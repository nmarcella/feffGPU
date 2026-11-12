# Agents and MCP

There are two ways for an agent (a language model that calls tools) to work with `feffgpu`:

1. **Using the software**: the package has an MCP server. MCP is the Model Context Protocol, the interface through
   which agents call tools. Through it an agent checks the installation, reads the documentation, writes and
   validates a configuration, starts a run, follows it, and reads results.
2. **Working on the code**: the repository has an `AGENTS.md` at its root with the setup, the test commands, a map
   of the modules, and the rules of the project. The documentation is also published as plain text for agents
   (`llms.txt` and `llms-full.txt` at the root of the documentation site).

## The MCP server

Install the extra and start the server on the machine that has the GPU:

```bash
pip install "feffgpu[mcp]"
feffgpu-mcp --workspace /path/to/project
```

The server speaks MCP over standard input and output, which is what desktop and command-line clients use. For a
remote machine (a cluster login node) it can serve HTTP:

```bash
feffgpu-mcp --workspace /path/to/project --transport streamable-http --host 127.0.0.1 --port 8765
```

It binds to this machine only by default. Reach it from elsewhere through an SSH tunnel. The server has no
authentication of its own: do not bind it to a public address.

### Client setup

Most MCP clients take a JSON entry of this form:

```json
{
  "mcpServers": {
    "feffgpu": {
      "command": "/path/to/venv/bin/feffgpu-mcp",
      "args": ["--workspace", "/path/to/project"]
    }
  }
}
```

In Claude Code:

```bash
claude mcp add feffgpu -- /path/to/venv/bin/feffgpu-mcp --workspace /path/to/project
```

Give the full path of the command inside the environment where `feffgpu` is installed.

### What the server may do

- It **reads and writes only inside the workspace** directory. A path outside it, also through a symbolic link, is
  refused.
- It **cannot run shell commands**. It starts only `feffgpu` commands from a fixed list.
- **Long runs are jobs.** A start tool returns an identifier at once. The run continues in the background, also
  after the client disconnects, and its record and log are kept in `.feffgpu_jobs/` inside the workspace.
- **One job per GPU at a time.** A job started while its GPU is busy waits in a queue.
- Results come back as **short summaries and file paths**, never as large arrays.

### Tools

| Tool | Does |
|---|---|
| `info` | versions, GPUs with free memory, the free-atom table, optional parts, problems |
| `list_docs`, `read_doc`, `search_docs` | the documentation pages (from a source checkout, or from the folder in `FEFFGPU_DOCS`) |
| `config_schema(kind)` | the JSON Schema of the configuration of MD-EXAFS (`"md"`) or RMC-EXAFS (`"rmc"`) |
| `example_config(kind, name)` | an example configuration as TOML text |
| `validate_config(kind, path)` | check a configuration file; problems are listed with the path of the field |
| `feff_run(feff_inp, out)` | start a FEFF calculation (a job) |
| `phase_build(feff_inp, out)` | start the calculation of a phase file (a job) |
| `md_exafs_start(config, shard)` | start MD-EXAFS (a job) |
| `rmc_start(config)` | start RMC-EXAFS (a job) |
| `command_start(tool, command, arguments)` | start another documented `md-exafs` or `rmc-exafs` subcommand (a job), for example a merge or an analysis |
| `job_status`, `job_list`, `job_log`, `job_cancel`, `job_result` | follow, list, read the output of, stop a job; list the files it wrote |
| `shell_table(structure, shells)` | N, R, MSRD and third cumulant of pair shells of a structure file, by integration |
| `short_pairs(structure, thresholds)` | pairs closer than given distances |
| `compare_spectrum(model, data, ...)` | the R-factor between a calculated and a measured chi(k), optionally with S0² and the E0 shift fitted |

Resources: `feffgpu://schema/md`, `feffgpu://schema/rmc` (the configuration schemas) and `feffgpu://docs/index`.

`feffgpu-mcp --list-tools` prints the tools with their descriptions as JSON.

### A typical session

1. `info`: is the installation complete?
2. `search_docs` / `read_doc("md/quickstart")`: how is the task done?
3. `example_config("md")`, then write a TOML file into the workspace with the client's own file tools.
4. `validate_config("md", "config.toml")` until it is valid.
5. `md_exafs_start("config.toml")` gives a job identifier.
6. `job_status(id)` from time to time; `job_log(id)` to see progress.
7. `job_result(id)` lists the output files; read `report.json` and the other documented files.

### From Python, without a client

The tools are plain methods and the job manager is a plain class:

```python
from feffgpu.mcp.server import Tools

tools = Tools("/path/to/project")
job = tools.feff_run("feff.inp", "chi.dat")
print(tools.job_status(job["id"]))
```

## Machine-readable interfaces

- Every command that prints a result has `--json`.
- Every configuration has a JSON Schema (`md-exafs schema`, `rmc-exafs schema`).
- Every output file has a documented schema ([](md/outputs.md), [](rmc/outputs.md)).

## Public services that read a repository

Some hosted services expose any public GitHub repository or its documentation to agents over MCP. They need no
setup on the side of this project once the repository is public. They give read access to code and documentation
only; running calculations needs the server above, on your own machine.
