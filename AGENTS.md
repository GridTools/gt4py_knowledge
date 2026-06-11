# gt4py Knowledge Base — Agent Instructions

This repository collects **design ideas and proposals** for
[gt4py](https://github.com/GridTools/gt4py). It is a low-barrier place to drop a
design, sketch, or proposal so that anyone — human or agent — can cross-check a
new idea against what already exists and spot overlaps or conflicts early.

It is **not** the gt4py source tree and **not** the formal ADR record. It is a
[Quartz](https://quartz.jzhao.xyz/) digital garden: all Markdown under
`content/` is published to GitHub Pages by `.github/workflows/deploy.yml` on
push to `main` (the workflow clones Quartz at build time — nothing is vendored
here). No local build is needed to author; just edit Markdown.

## Layout

```
content/
  index.md              # landing page = the keyworded, hierarchical index (see below)
  ideas/
    <person>/           # one subdirectory per contributor
      <proposal>.md           # a proposal/idea
      <proposal>_research.md  # optional appendix: background, research, prior art
      <proposal>_<topic>.md   # optional further appendices
      <proposal>/             # optional subdir: implementations illustrating the design
  accepted/             # proposals generally accepted but not yet implementation-ready
  templates/            # idea template (NOT published — see ignorePatterns)
```

- **`ideas/<person>/`** — your working area. Use your GitHub handle as the
  directory name. Filenames are free-form kebab-case slugs; no numbering.
- **`accepted/`** — flat directory of proposals the group broadly agrees on but
  that are not concrete enough to implement yet (may still have gaps). Entries
  here can be **removed** if flaws are discovered later — that is expected.
- An accepted idea that becomes concrete graduates to real work in gt4py (a PR,
  or a formal ADR in the gt4py repo); it can then be retired from here.

## Authoring a proposal

1. Copy `content/templates/idea.md` to `content/ideas/<your-handle>/<slug>.md`.
2. Fill the frontmatter:
   ```yaml
   ---
   title: Human-readable title
   author: <your-handle>
   tags: [keyword1, keyword2]   # the topics this document discusses
   created: 2026-06-11
   ---
   ```
3. Before writing, **skim the index and existing proposals** for overlap; link
   related/conflicting documents with `[[wikilinks]]` and call out the conflict
   explicitly. Surfacing conflicts is the whole point of this repo.
4. Cross-reference other notes with Obsidian-style `[[path/to/note|label]]`
   links — Quartz resolves them.
5. **Update `content/index.md`** (next section). This is required.

## Keep the index useful

`content/index.md` is the map of everything here and the first thing readers
and agents consult. It must stay current and keyword-rich:

- **Every** time you add, rename, move, or remove a document, update its index
  entry in the same change.
- Each entry is a wikilink plus a short **keywords** list naming the topics the
  document actually discusses — e.g.
  `- [[ideas/havogt/field-origin|Field origin rework]] — keywords: fields, domain, origin, embedded`.
  Keywords are what let people scan for overlapping ideas, so make them specific
  and honest about the content.
- Keep an entry's keywords **in sync with the document's `tags` frontmatter**
  (same vocabulary; Quartz also builds tag pages from `tags`).
- Group idea entries under a `### <person>` subsection of **Ideas**; nest
  appendices and implementation subdirs under their parent proposal.
- When a proposal is **accepted**, move the file from `ideas/<person>/` to
  `accepted/` and move its index entry from **Ideas** to **Accepted**, keeping
  the keywords. When an accepted proposal is **retired**, delete the file and
  its index entry.
- Prefer one consistent keyword vocabulary across entries (e.g. reuse `dace`,
  `unstructured`, `type-system`) so related ideas cluster and conflicts surface.

## Publishing notes

- `baseUrl` in `quartz.config.ts` must match the final GitHub Pages URL of this
  repo; update it if the repo moves.
- Anything under `templates/`, `private/`, or `.obsidian/` is excluded from the
  published site (`ignorePatterns`). Use `draft: true` in frontmatter to keep an
  in-progress note out of the published site while still committing it.
