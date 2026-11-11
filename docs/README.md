# Cerveau docs

Start with the [project README](../README.md): what Cerveau is, the results, and how to run it.

## Results

- [cerveau-scoreboard.html](cerveau-scoreboard.html): the results on one page, in plain language (download it and open it in a browser; GitHub shows it as code).
- [CERVEAU_LOCOMO_RESULTS.md](CERVEAU_LOCOMO_RESULTS.md): every LoCoMo run, the final grid, the human review of the grades, and the published numbers with why they don't compare directly.
- [LOCOMO_FINAL_RUNS.md](LOCOMO_FINAL_RUNS.md): how the final runs were planned (graders checked against hand labels, a frozen configuration, all questions).
- [PERSONAL_CHATS_EVAL.md](PERSONAL_CHATS_EVAL.md): the eval on the author's own chat history, the faults it found, and the model and setting comparisons.

## How it works

- [ARCHITECTURE.md](ARCHITECTURE.md): the design in full: memory model, consolidation, recall, activation, prompt budgets, schema.
- [GUIDE.md](GUIDE.md): a guided tour of the code, following one message and one consolidation end to end, then every module.
- [PIPELINE.md](PIPELINE.md): each step of a turn, by phase.

## Design notes, by feature

- [FACTS.md](FACTS.md): facts learned from conversations, deduplication and history.
- [ACTIVATION.md](ACTIVATION.md): follow-up questions, carry-over and the activation graph.
- [SEARCH_MEMORY_TOOL.md](SEARCH_MEMORY_TOOL.md): the model searching its own memory.
- [EMBEDDINGS.md](EMBEDDINGS.md): how vectors are stored, several models side by side, purge.
- [ATTACHMENTS.md](ATTACHMENTS.md): sharing files in the chat.
- [features/ENTITY_PROFILES.md](features/ENTITY_PROFILES.md): a person's facts as one block.
- [COMPACTION_EVAL.md](COMPACTION_EVAL.md): a proposed eval of what a running summary loses.

## The Django side

- [DJANGO_SETUP.md](DJANGO_SETUP.md): how Cerveau maps onto the Django starter template.
- [base_django_setup.md](base_django_setup.md): the template itself.

## The working log

- [STATUS.md](STATUS.md): what is built, every decision with its date, and the test list. Long: written as the project's day-to-day record.
- [github_workflow.md](github_workflow.md): the git rules the project follows.
