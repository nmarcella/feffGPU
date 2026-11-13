"""Triton kernels for the sequential radial recursions of FEFF's Dirac solver (FOVRG/intout.f and solin.f: the
Runge-Kutta start and the Milne predictor-corrector), one GPU thread per lane. Everything else of the solver (the
origin expansion, the flat-potential steps with their Bessel functions, the phase matching) stays in torch. The
arithmetic follows the torch version operation by operation; complex division is Smith's algorithm as in torch."""
from __future__ import annotations
import math
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                            # pragma: no cover
    HAVE_TRITON = False

# The Triton versions with which these kernels were compared with the torch code before a release
# (tests/test_kernels.py, feffgpu.xsph.kernel_difference: agreement to rounding). With one of them the kernels are used
# as they are. With any other version the comparison runs once per process before the kernels are used
# (feffgpu.xsph.kernels_usable). Add a version here only after tests/test_kernels.py passed with it on a GPU.
TESTED_TRITON = ("3.1.0", "3.8.0")

if HAVE_TRITON:

    @triton.jit
    def _cmul(ar, ai, br, bi):
        return ar * br - ai * bi, ar * bi + ai * br

    @triton.jit
    def _cdiv(ar, ai, br, bi):
        """(a / b) by Smith's algorithm (c10::complex operator/)."""
        abs_c = tl.abs(br)
        abs_d = tl.abs(bi)
        big = abs_c >= abs_d
        c1 = tl.where(big, br, bi)
        d1 = tl.where(big, bi, br)
        rat = d1 / c1
        scl = 1.0 / (c1 + d1 * rat)
        re_big = (ar + ai * rat) * scl
        im_big = (ai - ar * rat) * scl
        re_sml = (ar * rat + ai) * scl
        im_sml = (ai * rat - ar) * scl
        return tl.where(big, re_big, re_sml), tl.where(big, im_big, im_sml)

    @triton.jit
    def _cabs(ar, ai):
        return tl.sqrt(ar * ar + ai * ai)

    @triton.jit
    def _fg(ecr, eci, dvr, dvi, r, ccl):
        fr = (ecr - dvr) * r
        fi = (eci - dvi) * r
        return fr, fi, fr + ccl * r, fi

    @triton.jit
    def _c3(ic3, vmr, vmi, gr, gi):
        g2r, g2i = _cmul(gr, gi, gr, gi)
        return _cdiv(ic3 * vmr, ic3 * vmi, g2r, g2i)

    @triton.jit
    def milne_outward_kernel(ggr, ggi, gpr, gpi, dvr, dvi, vmr, vmi, dr, ecr, eci, kap, ic3, av1, iflat0, cst, L, N, nmax,
                             BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        hx = tl.load(cst + 0)
        exphx = tl.load(cst + 1)
        ccl = tl.load(cst + 2)
        a1 = tl.load(cst + 3)
        a2 = tl.load(cst + 4)
        a3 = tl.load(cst + 5)
        a4 = tl.load(cst + 6)
        a5 = tl.load(cst + 7)
        a6 = tl.load(cst + 8)
        test = tl.load(cst + 9)
        lane = pid * BLOCK + tl.arange(0, BLOCK)
        m = lane < L
        base = lane * N
        er = tl.load(ecr + lane, mask=m, other=0.0)
        ei = tl.load(eci + lane, mask=m, other=0.0)
        kp = tl.load(kap + lane, mask=m, other=1.0)
        c3f = tl.load(ic3 + lane, mask=m, other=0.0)
        a1v = tl.load(av1 + lane, mask=m, other=-1.0)
        ifl = tl.load(iflat0 + lane, mask=m, other=0)
        # ---- Runge-Kutta for the first NPI points
        g0r = tl.load(ggr + base, mask=m, other=0.0)
        g0i = tl.load(ggi + base, mask=m, other=0.0)
        p0r = tl.load(gpr + base, mask=m, other=0.0)
        p0i = tl.load(gpi + base, mask=m, other=0.0)
        r0 = tl.load(dr + base, mask=m, other=1.0)
        v0r = tl.load(dvr + base, mask=m, other=0.0)
        v0i = tl.load(dvi + base, mask=m, other=0.0)
        w0r = tl.load(vmr + base, mask=m, other=0.0)
        w0i = tl.load(vmi + base, mask=m, other=0.0)
        fr, fi, gr, gi = _fg(er, ei, v0r, v0i, r0, ccl)
        cr, ci = _c3(c3f, w0r, w0i, gr, gi)
        tr, ti = _cmul(gr, gi, p0r, p0i)
        d5r = hx * (tr - kp * g0r)
        d5i = hx * (ti - kp * g0i)
        tr, ti = _cmul(fr - cr, fi - ci, g0r, g0i)
        q5r = hx * (kp * p0r - tr)
        q5i = hx * (kp * p0i - ti)
        # d0..d5: the derivatives (times hx) at points 0..5; d5 is also the newest one while the loop runs. Each of
        # d1..d4 is written once, at its own step, with tl.where. They must not be filled by shifting variables that
        # start as the same value (d0 = d1, d1 = d2, ...): Triton 3.8 compiles that pattern wrongly (the shifted
        # variables keep their start value), where Triton 3.1 compiled it as written.
        d0r = d5r; d0i = d5i; q0r = q5r; q0i = q5i
        d1r = d5r; d1i = d5i; q1r = q5r; q1i = q5i
        d2r = d5r; d2i = d5i; q2r = q5r; q2i = q5i
        d3r = d5r; d3i = d5i; q3r = q5r; q3i = q5i
        d4r = d5r; d4i = d5i; q4r = q5r; q4i = q5i
        cgr = g0r; cgi = g0i; cpr = p0r; cpi = p0i                   # current point values
        for i in range(0, 5):
            ri_ = tl.load(dr + base + i, mask=m, other=1.0)
            rn = tl.load(dr + base + i + 1, mask=m, other=2.0)
            vir = tl.load(dvr + base + i, mask=m, other=0.0)
            vii = tl.load(dvi + base + i, mask=m, other=0.0)
            vnr = tl.load(dvr + base + i + 1, mask=m, other=0.0)
            vni = tl.load(dvi + base + i + 1, mask=m, other=0.0)
            wir = tl.load(vmr + base + i, mask=m, other=0.0)
            wii = tl.load(vmi + base + i, mask=m, other=0.0)
            wnr = tl.load(vmr + base + i + 1, mask=m, other=0.0)
            wni = tl.load(vmi + base + i + 1, mask=m, other=0.0)
            acr = cgr + 0.5 * d5r
            aci = cgi + 0.5 * d5i
            bcr = cpr + 0.5 * q5r
            bci = cpi + 0.5 * q5i
            rh = ri_ * exphx
            xm1 = (rn - rh) / (rn - ri_)
            xm2 = (rh - ri_) / (rn - ri_)
            dv1r = vir - a1v / ri_
            dv2r = vnr - a1v / rn
            vhr = dv1r * xm1 + dv2r * xm2
            vhi = vii * xm1 + vni * xm2
            vhr = vhr + a1v / rh
            vmhr = (xm1 * wir * ri_ + xm2 * wnr * rn) / rh
            vmhi = (xm1 * wii * ri_ + xm2 * wni * rn) / rh
            fr = (er - vhr) * rh
            fi = (ei - vhi) * rh
            gr = fr + ccl * rh
            gi = fi
            cr, ci = _c3(c3f, vmhr, vmhi, gr, gi)
            fcr = fr - cr
            fci = fi - ci
            tr, ti = _cmul(gr, gi, bcr, bci)
            dg2r = hx * (tr - kp * acr)
            dg2i = hx * (ti - kp * aci)
            tr, ti = _cmul(fcr, fci, acr, aci)
            dp2r = hx * (kp * bcr - tr)
            dp2i = hx * (kp * bci - ti)
            acr = acr + 0.5 * (dg2r - d5r)
            aci = aci + 0.5 * (dg2i - d5i)
            bcr = bcr + 0.5 * (dp2r - q5r)
            bci = bci + 0.5 * (dp2i - q5i)
            tr, ti = _cmul(gr, gi, bcr, bci)
            dg3r = hx * (tr - kp * acr)
            dg3i = hx * (ti - kp * aci)
            tr, ti = _cmul(fcr, fci, acr, aci)
            dp3r = hx * (kp * bcr - tr)
            dp3i = hx * (kp * bci - ti)
            acr = acr + dg3r - 0.5 * dg2r
            aci = aci + dg3i - 0.5 * dg2i
            bcr = bcr + dp3r - 0.5 * dp2r
            bci = bci + dp3i - 0.5 * dp2i
            fr, fi, gr, gi = _fg(er, ei, vnr, vni, rn, ccl)
            cr, ci = _c3(c3f, wnr, wni, gr, gi)
            fcr = fr - cr
            fci = fi - ci
            tr, ti = _cmul(gr, gi, bcr, bci)
            dg4r = hx * (tr - kp * acr)
            dg4i = hx * (ti - kp * aci)
            tr, ti = _cmul(fcr, fci, acr, aci)
            dp4r = hx * (kp * bcr - tr)
            dp4i = hx * (kp * bci - ti)
            ngr = cgr + (d5r + 2.0 * (dg2r + dg3r) + dg4r) / 6.0
            ngi = cgi + (d5i + 2.0 * (dg2i + dg3i) + dg4i) / 6.0
            npr = cpr + (q5r + 2.0 * (dp2r + dp3r) + dp4r) / 6.0
            npi = cpi + (q5i + 2.0 * (dp2i + dp3i) + dp4i) / 6.0
            tl.store(ggr + base + i + 1, ngr, mask=m)
            tl.store(ggi + base + i + 1, ngi, mask=m)
            tl.store(gpr + base + i + 1, npr, mask=m)
            tl.store(gpi + base + i + 1, npi, mask=m)
            tr, ti = _cmul(gr, gi, npr, npi)
            nd_r = hx * (tr - kp * ngr)
            nd_i = hx * (ti - kp * ngi)
            tr, ti = _cmul(fcr, fci, ngr, ngi)
            nq_r = hx * (kp * npr - tr)
            nq_i = hx * (kp * npi - ti)
            d1r = tl.where(i == 0, nd_r, d1r); d1i = tl.where(i == 0, nd_i, d1i)
            q1r = tl.where(i == 0, nq_r, q1r); q1i = tl.where(i == 0, nq_i, q1i)
            d2r = tl.where(i == 1, nd_r, d2r); d2i = tl.where(i == 1, nd_i, d2i)
            q2r = tl.where(i == 1, nq_r, q2r); q2i = tl.where(i == 1, nq_i, q2i)
            d3r = tl.where(i == 2, nd_r, d3r); d3i = tl.where(i == 2, nd_i, d3i)
            q3r = tl.where(i == 2, nq_r, q3r); q3i = tl.where(i == 2, nq_i, q3i)
            d4r = tl.where(i == 3, nd_r, d4r); d4i = tl.where(i == 3, nd_i, d4i)
            q4r = tl.where(i == 3, nq_r, q4r); q4i = tl.where(i == 3, nq_i, q4i)
            d5r = nd_r; d5i = nd_i; q5r = nq_r; q5i = nq_i
            cgr = ngr; cgi = ngi; cpr = npr; cpi = npi
        # after the loop d0..d5 hold the six derivatives (times hx) of points 0..5
        d0r = d0r / hx; d0i = d0i / hx; q0r = q0r / hx; q0i = q0i / hx
        d1r = d1r / hx; d1i = d1i / hx; q1r = q1r / hx; q1i = q1i / hx
        d2r = d2r / hx; d2i = d2i / hx; q2r = q2r / hx; q2i = q2i / hx
        d3r = d3r / hx; d3i = d3i / hx; q3r = q3r / hx; q3i = q3i / hx
        d4r = d4r / hx; d4i = d4i / hx; q4r = q4r / hx; q4i = q4i / hx
        d5r = d5r / hx; d5i = d5i / hx; q5r = q5r / hx; q5i = q5i / hx
        # ---- Milne predictor-corrector to iflat0
        for i in range(5, nmax):
            valid = (i + 1) <= ifl
            g5r = tl.load(ggr + base + i - 5, mask=m, other=0.0)
            g5i = tl.load(ggi + base + i - 5, mask=m, other=0.0)
            p5r = tl.load(gpr + base + i - 5, mask=m, other=0.0)
            p5i = tl.load(gpi + base + i - 5, mask=m, other=0.0)
            g3r = tl.load(ggr + base + i - 3, mask=m, other=0.0)
            g3i = tl.load(ggi + base + i - 3, mask=m, other=0.0)
            p3r = tl.load(gpr + base + i - 3, mask=m, other=0.0)
            p3i = tl.load(gpi + base + i - 3, mask=m, other=0.0)
            acpr = g5r + a1 * (d5r + d1r) + a2 * (d4r + d2r) + a3 * d3r
            acpi = g5i + a1 * (d5i + d1i) + a2 * (d4i + d2i) + a3 * d3i
            bcpr = p5r + a1 * (q5r + q1r) + a2 * (q4r + q2r) + a3 * q3r
            bcpi = p5i + a1 * (q5i + q1i) + a2 * (q4i + q2i) + a3 * q3i
            acr = g3r + a4 * d2r + a5 * (d5r + d3r) + a6 * d4r
            aci = g3i + a4 * d2i + a5 * (d5i + d3i) + a6 * d4i
            bcr = p3r + a4 * q2r + a5 * (q5r + q3r) + a6 * q4r
            bci = p3i + a4 * q2i + a5 * (q5i + q3i) + a6 * q4i
            rn = tl.load(dr + base + i + 1, mask=m, other=1.0)
            vnr = tl.load(dvr + base + i + 1, mask=m, other=0.0)
            vni = tl.load(dvi + base + i + 1, mask=m, other=0.0)
            wnr = tl.load(vmr + base + i + 1, mask=m, other=0.0)
            wni = tl.load(vmi + base + i + 1, mask=m, other=0.0)
            fr, fi, gr, gi = _fg(er, ei, vnr, vni, rn, ccl)
            cr, ci = _c3(c3f, wnr, wni, gr, gi)
            fcr = fr - cr
            fci = fi - ci
            tr, ti = _cmul(gr, gi, bcpr, bcpi)
            dg6r = tr - kp * acpr
            dg6i = ti - kp * acpi
            tr, ti = _cmul(fcr, fci, acpr, acpi)
            dp6r = kp * bcpr - tr
            dp6i = kp * bcpi - ti
            ggnr = acr + a4 * dg6r
            ggni = aci + a4 * dg6i
            gpnr = bcr + a4 * dp6r
            gpni = bci + a4 * dp6i
            fail = ((_cabs(test * (ggnr - acpr), test * (ggni - acpi)) > _cabs(ggnr, ggni)) |
                    (_cabs(test * (gpnr - bcpr), test * (gpni - bcpi)) > _cabs(gpnr, gpni))) & valid & m
            nit = 0
            nfail = tl.sum(fail.to(tl.int32), 0)
            while (nfail > 0) & (nit < 40):
                acpr = tl.where(fail, ggnr, acpr)
                acpi = tl.where(fail, ggni, acpi)
                bcpr = tl.where(fail, gpnr, bcpr)
                bcpi = tl.where(fail, gpni, bcpi)
                nit += 1
                tr, ti = _cmul(gr, gi, bcpr, bcpi)
                dg6nr = tr - kp * acpr
                dg6ni = ti - kp * acpi
                tr, ti = _cmul(fcr, fci, acpr, acpi)
                dp6nr = kp * bcpr - tr
                dp6ni = kp * bcpi - ti
                dg6r = tl.where(fail, dg6nr, dg6r)
                dg6i = tl.where(fail, dg6ni, dg6i)
                dp6r = tl.where(fail, dp6nr, dp6r)
                dp6i = tl.where(fail, dp6ni, dp6i)
                ggnr = tl.where(fail, acr + a4 * dg6nr, ggnr)
                ggni = tl.where(fail, aci + a4 * dg6ni, ggni)
                gpnr = tl.where(fail, bcr + a4 * dp6nr, gpnr)
                gpni = tl.where(fail, bci + a4 * dp6ni, gpni)
                fail = ((_cabs(test * (ggnr - acpr), test * (ggni - acpi)) > _cabs(ggnr, ggni)) |
                        (_cabs(test * (gpnr - bcpr), test * (gpni - bcpi)) > _cabs(gpnr, gpni))) & fail
                nfail = tl.sum(fail.to(tl.int32), 0)
            zero = ggnr * 0.0
            tl.store(ggr + base + i + 1, tl.where(valid, ggnr, zero), mask=m)
            tl.store(ggi + base + i + 1, tl.where(valid, ggni, zero), mask=m)
            tl.store(gpr + base + i + 1, tl.where(valid, gpnr, zero), mask=m)
            tl.store(gpi + base + i + 1, tl.where(valid, gpni, zero), mask=m)
            d0r = d1r; d0i = d1i; q0r = q1r; q0i = q1i
            d1r = d2r; d1i = d2i; q1r = q2r; q1i = q2i
            d2r = d3r; d2i = d3i; q2r = q3r; q2i = q3i
            d3r = d4r; d3i = d4i; q3r = q4r; q3i = q4i
            d4r = d5r; d4i = d5i; q4r = q5r; q4i = q5i
            d5r = tl.where(valid, dg6r, zero); d5i = tl.where(valid, dg6i, zero)
            q5r = tl.where(valid, dp6r, zero); q5i = tl.where(valid, dp6i, zero)

    @triton.jit
    def milne_inward_kernel(ggr, ggi, gpr, gpi, Dr, Di, Pr, Pi, dvr, dvi, vmr, vmi, dr, ecr, eci, kap, ic3, iflat0, cst, L, N, nmax,
                            BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        hx = tl.load(cst + 0)
        ccl = tl.load(cst + 2)
        a1 = tl.load(cst + 3)
        a2 = tl.load(cst + 4)
        a3 = tl.load(cst + 5)
        a4 = tl.load(cst + 6)
        a5 = tl.load(cst + 7)
        a6 = tl.load(cst + 8)
        test = tl.load(cst + 9)
        lane = pid * BLOCK + tl.arange(0, BLOCK)
        m = lane < L
        base = lane * N
        b6 = lane * 6
        er = tl.load(ecr + lane, mask=m, other=0.0)
        ei = tl.load(eci + lane, mask=m, other=0.0)
        kp = tl.load(kap + lane, mask=m, other=1.0)
        c3f = tl.load(ic3 + lane, mask=m, other=0.0)
        ifl = tl.load(iflat0 + lane, mask=m, other=0)
        d0r = tl.load(Dr + b6 + 0, mask=m, other=0.0); d0i = tl.load(Di + b6 + 0, mask=m, other=0.0)
        d1r = tl.load(Dr + b6 + 1, mask=m, other=0.0); d1i = tl.load(Di + b6 + 1, mask=m, other=0.0)
        d2r = tl.load(Dr + b6 + 2, mask=m, other=0.0); d2i = tl.load(Di + b6 + 2, mask=m, other=0.0)
        d3r = tl.load(Dr + b6 + 3, mask=m, other=0.0); d3i = tl.load(Di + b6 + 3, mask=m, other=0.0)
        d4r = tl.load(Dr + b6 + 4, mask=m, other=0.0); d4i = tl.load(Di + b6 + 4, mask=m, other=0.0)
        d5r = tl.load(Dr + b6 + 5, mask=m, other=0.0); d5i = tl.load(Di + b6 + 5, mask=m, other=0.0)
        q0r = tl.load(Pr + b6 + 0, mask=m, other=0.0); q0i = tl.load(Pi + b6 + 0, mask=m, other=0.0)
        q1r = tl.load(Pr + b6 + 1, mask=m, other=0.0); q1i = tl.load(Pi + b6 + 1, mask=m, other=0.0)
        q2r = tl.load(Pr + b6 + 2, mask=m, other=0.0); q2i = tl.load(Pi + b6 + 2, mask=m, other=0.0)
        q3r = tl.load(Pr + b6 + 3, mask=m, other=0.0); q3i = tl.load(Pi + b6 + 3, mask=m, other=0.0)
        q4r = tl.load(Pr + b6 + 4, mask=m, other=0.0); q4i = tl.load(Pi + b6 + 4, mask=m, other=0.0)
        q5r = tl.load(Pr + b6 + 5, mask=m, other=0.0); q5i = tl.load(Pi + b6 + 5, mask=m, other=0.0)
        for ii in range(0, nmax):
            i = nmax - ii                                         # i = nmax .. 1, computing point i - 1
            valid = (i <= ifl) & m
            g5r = tl.load(ggr + base + i + 5, mask=m, other=0.0)
            g5i = tl.load(ggi + base + i + 5, mask=m, other=0.0)
            p5r = tl.load(gpr + base + i + 5, mask=m, other=0.0)
            p5i = tl.load(gpi + base + i + 5, mask=m, other=0.0)
            g3r = tl.load(ggr + base + i + 3, mask=m, other=0.0)
            g3i = tl.load(ggi + base + i + 3, mask=m, other=0.0)
            p3r = tl.load(gpr + base + i + 3, mask=m, other=0.0)
            p3i = tl.load(gpi + base + i + 3, mask=m, other=0.0)
            acpr = g5r + a1 * (d5r + d1r) + a2 * (d4r + d2r) + a3 * d3r
            acpi = g5i + a1 * (d5i + d1i) + a2 * (d4i + d2i) + a3 * d3i
            bcpr = p5r + a1 * (q5r + q1r) + a2 * (q4r + q2r) + a3 * q3r
            bcpi = p5i + a1 * (q5i + q1i) + a2 * (q4i + q2i) + a3 * q3i
            acr = g3r + a4 * d2r + a5 * (d5r + d3r) + a6 * d4r
            aci = g3i + a4 * d2i + a5 * (d5i + d3i) + a6 * d4i
            bcr = p3r + a4 * q2r + a5 * (q5r + q3r) + a6 * q4r
            bci = p3i + a4 * q2i + a5 * (q5i + q3i) + a6 * q4i
            rp = tl.load(dr + base + i - 1, mask=m, other=1.0)
            vpr = tl.load(dvr + base + i - 1, mask=m, other=0.0)
            vpi = tl.load(dvi + base + i - 1, mask=m, other=0.0)
            wpr = tl.load(vmr + base + i - 1, mask=m, other=0.0)
            wpi = tl.load(vmi + base + i - 1, mask=m, other=0.0)
            fr, fi, gr, gi = _fg(er, ei, vpr, vpi, rp, ccl)
            cr, ci = _c3(c3f, wpr, wpi, gr, gi)
            fcr = fr - cr
            fci = fi - ci
            tr, ti = _cmul(gr, gi, bcpr, bcpi)
            dg6r = -(tr - kp * acpr)
            dg6i = -(ti - kp * acpi)
            tr, ti = _cmul(fcr, fci, acpr, acpi)
            dp6r = -(kp * bcpr - tr)
            dp6i = -(kp * bcpi - ti)
            ggnr = acr + a4 * dg6r
            ggni = aci + a4 * dg6i
            gpnr = bcr + a4 * dp6r
            gpni = bci + a4 * dp6i
            fail = ((_cabs(test * (ggnr - acpr), test * (ggni - acpi)) > _cabs(ggnr, ggni)) |
                    (_cabs(test * (gpnr - bcpr), test * (gpni - bcpi)) > _cabs(gpnr, gpni))) & valid
            nit = 0
            nfail = tl.sum(fail.to(tl.int32), 0)
            while (nfail > 0) & (nit < 40):
                acpr = tl.where(fail, ggnr, acpr)
                acpi = tl.where(fail, ggni, acpi)
                bcpr = tl.where(fail, gpnr, bcpr)
                bcpi = tl.where(fail, gpni, bcpi)
                nit += 1
                tr, ti = _cmul(gr, gi, bcpr, bcpi)
                dg6nr = -(tr - kp * acpr)
                dg6ni = -(ti - kp * acpi)
                tr, ti = _cmul(fcr, fci, acpr, acpi)
                dp6nr = -(kp * bcpr - tr)
                dp6ni = -(kp * bcpi - ti)
                dg6r = tl.where(fail, dg6nr, dg6r)
                dg6i = tl.where(fail, dg6ni, dg6i)
                dp6r = tl.where(fail, dp6nr, dp6r)
                dp6i = tl.where(fail, dp6ni, dp6i)
                ggnr = tl.where(fail, acr + a4 * dg6nr, ggnr)
                ggni = tl.where(fail, aci + a4 * dg6ni, ggni)
                gpnr = tl.where(fail, bcr + a4 * dp6nr, gpnr)
                gpni = tl.where(fail, bci + a4 * dp6ni, gpni)
                fail = ((_cabs(test * (ggnr - acpr), test * (ggni - acpi)) > _cabs(ggnr, ggni)) |
                        (_cabs(test * (gpnr - bcpr), test * (gpni - bcpi)) > _cabs(gpnr, gpni))) & fail
                nfail = tl.sum(fail.to(tl.int32), 0)
            oldr = tl.load(ggr + base + i - 1, mask=m, other=0.0)
            oldi = tl.load(ggi + base + i - 1, mask=m, other=0.0)
            olpr = tl.load(gpr + base + i - 1, mask=m, other=0.0)
            olpi = tl.load(gpi + base + i - 1, mask=m, other=0.0)
            tl.store(ggr + base + i - 1, tl.where(valid, ggnr, oldr), mask=m)
            tl.store(ggi + base + i - 1, tl.where(valid, ggni, oldi), mask=m)
            tl.store(gpr + base + i - 1, tl.where(valid, gpnr, olpr), mask=m)
            tl.store(gpi + base + i - 1, tl.where(valid, gpni, olpi), mask=m)
            n0r = tl.where(valid, d1r, d0r); n0i = tl.where(valid, d1i, d0i)
            n1r = tl.where(valid, d2r, d1r); n1i = tl.where(valid, d2i, d1i)
            n2r = tl.where(valid, d3r, d2r); n2i = tl.where(valid, d3i, d2i)
            n3r = tl.where(valid, d4r, d3r); n3i = tl.where(valid, d4i, d3i)
            n4r = tl.where(valid, d5r, d4r); n4i = tl.where(valid, d5i, d4i)
            n5r = tl.where(valid, dg6r, d5r); n5i = tl.where(valid, dg6i, d5i)
            m0r = tl.where(valid, q1r, q0r); m0i = tl.where(valid, q1i, q0i)
            m1r = tl.where(valid, q2r, q1r); m1i = tl.where(valid, q2i, q1i)
            m2r = tl.where(valid, q3r, q2r); m2i = tl.where(valid, q3i, q2i)
            m3r = tl.where(valid, q4r, q3r); m3i = tl.where(valid, q4i, q3i)
            m4r = tl.where(valid, q5r, q4r); m4i = tl.where(valid, q5i, q4i)
            m5r = tl.where(valid, dp6r, q5r); m5i = tl.where(valid, dp6i, q5i)
            d0r = n0r; d0i = n0i; d1r = n1r; d1i = n1i; d2r = n2r; d2i = n2i; d3r = n3r; d3i = n3i; d4r = n4r; d4i = n4i; d5r = n5r; d5i = n5i
            q0r = m0r; q0i = m0i; q1r = m1r; q1i = m1i; q2r = m2r; q2i = m2i; q3r = m3r; q3i = m3i; q4r = m4r; q4i = m4i; q5r = m5r; q5i = m5i


def _split(z):
    r = torch.view_as_real(z.contiguous())
    return r[..., 0].contiguous(), r[..., 1].contiguous()


def milne_outward(gg, gp, dv, vm, dr, ec, kap, ic3, av1, iflat0, hx, consts, BLOCK=64):
    """Fill gg, gp (L, N) complex128 (point 0 set) up to iflat0 per lane. consts: dict with ccl, a1..a6, test."""
    L, N = gg.shape
    ggr, ggi = _split(gg)
    gpr, gpi = _split(gp)
    dvr, dvi = _split(dv)
    vmr, vmi = _split(vm)
    ecr, eci = _split(ec)
    nmax = int(iflat0.max())
    grid = ((L + BLOCK - 1) // BLOCK,)
    cst = torch.tensor([hx, math.exp(hx / 2), consts["ccl"], consts["a1"], consts["a2"], consts["a3"], consts["a4"], consts["a5"],
                        consts["a6"], consts["test"]], dtype=torch.float64, device=gg.device)
    milne_outward_kernel[grid](ggr, ggi, gpr, gpi, dvr, dvi, vmr, vmi, dr.contiguous(), ecr, eci, kap.contiguous(), ic3.contiguous(),
                               av1.contiguous(), iflat0.contiguous(), cst, L, N, nmax, BLOCK=BLOCK)
    return torch.complex(ggr, ggi), torch.complex(gpr, gpi)


def milne_inward(gg, gp, D, P, dv, vm, dr, ec, kap, ic3, iflat0, hx, consts, BLOCK=64):
    """Inward Milne from iflat0 to the origin, gg/gp (L, N) set from iflat0 outward, D/P (L, 6) seeds."""
    L, N = gg.shape
    ggr, ggi = _split(gg)
    gpr, gpi = _split(gp)
    Dr, Di = _split(D)
    Pr, Pi = _split(P)
    dvr, dvi = _split(dv)
    vmr, vmi = _split(vm)
    ecr, eci = _split(ec)
    nmax = int(iflat0.max())
    grid = ((L + BLOCK - 1) // BLOCK,)
    cst = torch.tensor([hx, math.exp(hx / 2), consts["ccl"], consts["a1"], consts["a2"], consts["a3"], consts["a4"], consts["a5"],
                        consts["a6"], consts["test"]], dtype=torch.float64, device=gg.device)
    milne_inward_kernel[grid](ggr, ggi, gpr, gpi, Dr, Di, Pr, Pi, dvr, dvi, vmr, vmi, dr.contiguous(), ecr, eci, kap.contiguous(),
                              ic3.contiguous(), iflat0.contiguous(), cst, L, N, nmax, BLOCK=BLOCK)
    return torch.complex(ggr, ggi), torch.complex(gpr, gpi)
