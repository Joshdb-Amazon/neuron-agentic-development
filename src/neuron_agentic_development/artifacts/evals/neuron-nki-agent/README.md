# CCNKIDevSuitePlugin

A Claude Code plugin for developing NKI (Neuron Kernel Interface) kernels on AWS Trainium and Inferentia, along with evaluation data measuring plugin effectiveness.

## Repository Structure

```
CCNKIDevSuitePlugin/
├── plugin/                     # The Claude Code plugin
│   ├── .claude-plugin/         # Plugin manifest
│   ├── .mcp.json               # MCP server configuration
│   ├── agents/                 # Autonomous agent definitions
│   ├── hooks/                  # Environment validation hooks
│   ├── skills/                 # User-invocable skills (writing, debugging, profiling, etc.)
│   ├── Config                  # Build configuration
│   └── README.md               # Detailed plugin documentation
│
├── evaluation/                 # Evaluation data and example runs
│   ├── benchmarks/             # Formal A/B evaluation (with vs without plugin)
│   │   ├── evals.json          # 17 eval definitions with prompts and rubrics
│   │   └── EVAL_REPORT.md      # Results, analysis, and cost breakdown
│   ├── examples/               # End-to-end demonstration runs
│   │   └── <kernel>/           # One directory per kernel
│   │       ├── prompt.md       # Task description
│   │       ├── <kernel>.py     # PyTorch reference implementation
│   │       └── output/         # Generated NKI kernel, tests, and findings
│   └── README.md               # Explains both collections and their relationship
│
├── .gitignore
└── README.md                   # This file
```

### plugin/

Contains the full Claude Code plugin: skills, agents, hooks, and configuration. This is the directory you point Claude Code at to enable the plugin. See [plugin/README.md](plugin/README.md) for detailed documentation on available skills, agents, and configuration options.

### evaluation/

Contains two complementary collections measuring plugin effectiveness. **Benchmarks** (`benchmarks/`) provide a formal A/B comparison of 17 tasks run with and without the plugin — the key finding is 100% Beta 2 API compliance with the plugin vs 0% without. **Examples** (`examples/`) are 8 end-to-end demonstration runs producing NKI kernels from PyTorch code, with device-verified correctness. Some kernels (softmax, gelu, matmul) appear in both collections with different specs. See [evaluation/README.md](evaluation/README.md) for details.

## Enabling the Plugin

To use the NKI Dev Suite plugin with Claude Code, run:

```bash
claude --plugin-dir /path/to/CCNKIDevSuitePlugin/plugin
```

For example, if you cloned this repo to `~/projects/CCNKIDevSuitePlugin`:

```bash
claude --plugin-dir ~/projects/CCNKIDevSuitePlugin/plugin
```

### Configuration

After enabling the plugin, create `.claude/nki-dev-suite.local.md` in your working directory with your settings:

```yaml
---
nki_venv_path: "/path/to/neuronx-venv"
default_nc_version: "gen4"
---
```

Or set environment variables:
- `NKI_VENV_PATH` - Path to Python venv with neuronx packages
- `NKI_NC_VERSION` - Hardware target (gen2/gen3/gen4)

## License

MIT
