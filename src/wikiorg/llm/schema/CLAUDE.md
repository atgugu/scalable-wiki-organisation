# Wiki maintainer schema
You maintain a personal wiki built from the documents in `raw/`. You are its only editor. Follow this file exactly.

## Layout
- `raw/<doc_id>.md` — source documents. READ-ONLY: never create, edit or delete anything in `raw/`.
- `wiki/sources/<doc_id>.md` — one summary page per source (same file stem as the raw file).
- `wiki/topics/<slug>.md` — topic pages, the categories of this wiki. Every source page has exactly one primary topic.
- `wiki/entities/<slug>.md` — pages about specific people, organisations, places, works, events.
- `wiki/concepts/<slug>.md` — pages about general ideas, methods, phenomena, classes of things.
- `wiki/index.md` — catalogue of every wiki page, by category (Topics, Sources, Entities, Concepts).
- `wiki/log.md` — append-only chronological log of every ingest and lint.
- `wiki/overview.md` — short synthesis of what the wiki covers (at most 400 words).
Slugs: lowercase ASCII words joined by hyphens (`association-football`, `marie-curie`). One page per real-world thing:
check for an existing page (spelling variants, singular/plural, abbreviations) before creating one.

## Links
- Relative Markdown links only: from `wiki/sources/x.md` write `[Title](../topics/slug.md)`; from `wiki/index.md` write `[Title](topics/slug.md)`.
- Link only to files that exist at the moment you write the link (check with Glob). Never link to a page you have not created.
- No `[[wiki-links]]`, no URLs, no links into `raw/` other than the `source:` field.

## Page templates
Source page `wiki/sources/<doc_id>.md`:

    ---
    title: "<exact title from the raw file>"
    type: source
    source: raw/<doc_id>.md
    topic: <topic-slug>            # exactly one; must be an existing file wiki/topics/<topic-slug>.md
    related: [<doc_id>, <doc_id>]  # 0-5 existing source pages, most related first, never this page itself
    entities: [<slug>, ...]        # entity pages linked from this page
    concepts: [<slug>, ...]        # concept pages linked from this page
    ingest: <n>                    # ingest number (count of entries in log.md including this one)
    ---
    # <Title>
    120-250 words of plain prose stating what the source says (facts, names, dates, numbers). No commentary.

    ## Related
    - [Title](<doc_id>.md) — one clause on why (same pages and order as `related:`)

Topic page `wiki/topics/<slug>.md`:

    ---
    title: "<Topic label>"   # noun phrase of at most 4 words at the granularity of an encyclopedia category, e.g. "Rivers of Europe", "Video games", "Politicians"
    type: topic
    ---
    # <Topic label>
    Scope: 2-3 sentences saying what belongs here.

    ## Sources
    - [Title](../sources/<doc_id>.md) — one-line gloss

    ## Related topics
    - [Label](<slug>.md)

Entity page `wiki/entities/<slug>.md` and concept page `wiki/concepts/<slug>.md`:

    ---
    title: "<Name>"
    type: entity            # or: concept
    ---
    # <Name>
    2-4 sentences synthesised from the sources.

    ## Facts
    - <one fact> ([Title](../sources/<doc_id>.md))

    ## Mentioned in
    - [Title](../sources/<doc_id>.md)

`wiki/index.md`: sections `## Topics`, `## Sources`, `## Entities`, `## Concepts`; one bullet per page, alphabetical within a section:
`- [Label](topics/slug.md) — <n> sources` / `- [Title](sources/id.md) — [Label](topics/slug.md)` / `- [Name](entities/slug.md) — 3-8 word gloss`.
`wiki/log.md`: append `## [YYYY-MM-DD] ingest | <Title>` followed by one line `created: <paths>; updated: <paths>`.
Lint entries: `## [YYYY-MM-DD] lint | <n> pages checked` followed by one line per change.

## Ingest procedure (prompt: "Ingest raw/<doc_id>.md following CLAUDE.md.")
1. Read `raw/<doc_id>.md` in full, then read `wiki/index.md`.
2. Choose the primary topic. Reuse an existing topic page whenever the source fits its scope; create a new topic page only
   when no existing topic fits. A topic is a category, never a single entity; never create a near-duplicate of an existing topic.
3. Identify up to 5 entities and up to 3 concepts central to the source; for each, find the existing page (index, Glob) or plan a new one.
4. Choose up to 5 related source pages, most related first, from the topic page's Sources list and the index; you may read at
   most 5 candidate source pages to decide. If no source page exists yet, `related:` is empty.
5. Write `wiki/sources/<doc_id>.md` from the template.
6. Create or update the topic page (add the source under Sources; adjust Scope only if needed).
7. Create or update each entity and concept page: add the source under Mentioned in and at most 3 new facts per page.
   Edit existing text only to fix an error or merge a duplicate.
8. Update `wiki/index.md` (new pages, topic counts) and append the `wiki/log.md` entry.
9. Check that every link you wrote resolves to an existing file (Glob) and that `topic:` and `related:` match the page body.
10. Reply with exactly one line: `DONE topic=<slug> new_topic=<yes|no> files=<number of files under wiki/ created or modified>`. Nothing else.

Limits per ingest: at most 15 files under `wiki/` created or modified, index.md and log.md included. If more candidates
exist, prioritise: source page > topic page > index > log > entities > concepts. Touch `overview.md` only when you created a
new topic (add one line for it).

## Lint procedure (prompt: "Run the lint procedure in CLAUDE.md.")
1. Broken links: Grep every `](` link under `wiki/`; fix or remove any whose target does not exist.
2. Orphans: every page must be listed in `wiki/index.md` and every topic/entity/concept page must have an inbound link from a source page; add the missing links.
3. Duplicates: merge topic, entity or concept pages about the same thing (keep the better slug, move content, update every link, drop the duplicate from the index).
4. Contradictions and stale claims: when two pages state incompatible facts, record both with their sources under `## Open questions` on that page.
5. Missing pages: create an entity or concept page for anything named by 3 or more source pages that has no page yet.
6. Rewrite `wiki/overview.md` (at most 400 words): what the wiki covers, the main topics with links, notable groups of related sources.
7. Append the log entry and reply with one line: `DONE lint files=<number of files created, modified or deleted>`.

Limits per lint: read at most 60 pages, change at most 40 files.

## Writing rules
- Plain, dense, factual prose. No filler, no meta-commentary, no restating this schema.
- Prefer Edit on existing pages; never rewrite a page when a small edit suffices; do not re-read a file already read in this session.
- Use no information other than `raw/` and `wiki/`; never invent facts. Never modify `raw/` or `CLAUDE.md`; never create files outside `wiki/`.
