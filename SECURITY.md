# Security

## Reporting

Please report a security problem privately through GitHub's "Report a vulnerability" function of this repository
(Security tab), not in a public issue. Describe the kind of problem and how to observe it.

## Things to know

- **Phase files written by `feffgpu phase` are Python pickles.** Loading a pickle runs code. Load only phase files
  that you made yourself or that come from someone you trust. FEFF's `phase.pad` is a plain data file and can be
  used instead.
- **The MCP server** (`feffgpu-mcp`) reads and writes only inside its workspace directory and starts only a fixed
  list of `feffgpu` commands. It has no authentication of its own: with an HTTP transport, keep it bound to
  `127.0.0.1` and reach it through an SSH tunnel.
- **The free-atom table** is downloaded over HTTPS and checked against a SHA-256 checksum that is part of the
  package.
