# Ethics Note

Corpus: r/gradadmissions with 27,219 posts, 111,212 comments, and 27,169 stored user records from 2025-09-24 to 2026-04-04.

## Personal Information and Re-identification

Even if usernames are treated as pseudonyms, admissions posts often contain combinations of GPA, university names, target programs, research fields, publications, visa status, timelines, and admit/reject outcomes. Those attributes can be enough to re-identify a real applicant when combined with posting history or external profiles. The RAG system therefore should avoid revealing usernames in answers, should cite content at the snippet level rather than profile users, and should not answer identity-seeking questions.

## Right to be Forgotten

A local archived RAG system can violate deletion expectations if a user deletes a Reddit post after it has already been ingested and embedded. Full compliance in production is difficult but the design should include deletion sync, tombstones for removed Reddit IDs, vector-index rebuilds or targeted vector deletion, audit logs, and a documented retention policy. This course implementation does not continuously poll Reddit for deletions, so it should be presented as a research prototype rather than a production system.

## Practical Safeguards

The assistant prompt instructs models to use only retrieved context, refuse unsupported or identity-seeking questions, avoid private identity inference, and summarize community-level patterns instead of profiling individual users. Stored API keys remain in `.env`, and generated reports should not include secrets.
