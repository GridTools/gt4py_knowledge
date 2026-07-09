# gt4py Knowledge Base

A low-barrier collection of **design ideas and proposals** for
[gt4py](https://github.com/GridTools/gt4py), published as a
[Quartz](https://quartz.jzhao.xyz/) digital garden to GitHub Pages.

Drop a proposal, cross-check it against existing ones, and surface conflicts
early. This is neither the gt4py source tree nor the formal ADR record.

See [`AGENTS.md`](AGENTS.md) for the structure, authoring workflow, and the rules
for keeping [`content/index.md`](content/index.md) useful.

Every proposal must include a `status` field in its frontmatter: `draft`
(default), `reviewed`, or `final`.

## Structure

```
content/
  index.md              # landing page = keyworded, hierarchical index
  personal/
    <person>/                   # one subdirectory per contributor
      <proposal>.md             # a single-file proposal/idea
      <proposal>/               # a multi-file proposal/idea (REQUIRED if >1 file)
        <proposal>.md           # the main proposal/idea
        <proposal>_research.md  # optional appendix: background, research, prior art
        <proposal>_<topic>.md   # optional further appendices
  shared/               # proposals that are discussed by the team (only touch with PR review)
  templates/            # idea template (NOT published — see ignorePatterns)
quartz.config.ts        # Quartz config (set baseUrl to the Pages URL)
quartz.layout.ts        # Quartz layout
.github/workflows/deploy.yml   # build (clone Quartz + copy content) & deploy to Pages
```

## Setup (one-time, after the repo lands on GitHub)

1. Set `baseUrl` in `quartz.config.ts` to `<owner>.github.io/<repo>`.
2. Repo **Settings → Pages → Source → GitHub Actions**.
3. Push to `main`; the workflow builds and deploys automatically.

## Local preview

```bash
git clone --depth 1 --branch v4 https://github.com/jackyzha0/quartz.git /tmp/quartz-preview
cd /tmp/quartz-preview && npm ci
# from this repo's root:
rm -rf /tmp/quartz-preview/content/* && cp -r content/* /tmp/quartz-preview/content/
cp quartz.config.ts quartz.layout.ts /tmp/quartz-preview/
cd /tmp/quartz-preview && npx quartz build --serve   # http://localhost:8080
```
