"""Agent access through MCP (the Model Context Protocol): a server whose tools validate configurations, start FEFF,
MD-EXAFS and RMC-EXAFS runs as jobs, follow them, and read results and documentation.

``feffgpu.mcp.server.Tools`` holds the tools as plain methods (usable without an MCP client);
``feffgpu.mcp.jobs.JobManager`` runs ``feffgpu`` commands as background jobs inside a workspace directory.
The MCP SDK is an optional dependency: ``pip install 'feffgpu[mcp]'``.
"""
from .jobs import JobManager, Workspace, WorkspaceError

__all__ = ["JobManager", "Workspace", "WorkspaceError"]
