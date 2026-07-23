# Project Memory Bank

## Architecture Philosophy
- We are building a modular application. Each major feature must be separated into its own directory or file.
- Do not put multiple distinct features or massive components into a single file.

## File Structure & Guidelines
- Domain-specific logic, styles, and routes should live in individual files (e.g., separate files for `auth`, `dashboard`, `database` configuration, etc.).
- Always check the `docs/architecture.md` file to see where new features should be placed.
- When creating a new feature, first update `docs/tasks.md` with the implementation steps, then build it one file at a time.

## Coding Standards
- Use modular imports to connect these separate files.
- Never use placeholders. Always provide full, runnable code blocks.
- Keep components small, focused, and reusable.
