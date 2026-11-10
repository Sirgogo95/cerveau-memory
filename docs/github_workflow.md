# GitHub Workflow

This document defines the mandatory git workflow for this repository. The rules below MUST be followed for every git operation performed in this repo, by humans and AI agents alike. They are not suggestions.

## Git Workflow Rules (MANDATORY)

### Protected Branches

- **`main`**: Holds production-ready code only. NEVER commit directly to `main`. NEVER push to `main`. Changes reach `main` only through merges from `development`.
- **`development`**: Integration branch where features are combined and tested before merging into `main`. Do not commit experimental or unfinished work directly here; use a working branch and merge in.

If you are asked to make code changes while on `main` or `development`, create an appropriately named working branch FIRST (see below), then commit there.

### Branch Naming (MANDATORY)

Every new branch MUST use one of these three prefixes:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/*` | Adding a new feature to the project | `feature/new-feature-description` |
| `fix/*` | Hotfixes or bug fixes | `fix/bug-description` |
| `sandbox/*` | Experimentation and testing new ideas | `sandbox/experiment-feature` |

Branch name formatting rules, ALL of which must hold:

1. All lowercase. No uppercase letters anywhere.
2. Use hyphens (`-`) instead of spaces. Never use underscores or spaces.
3. No special characters except the single `/` that separates the type prefix from the description.
4. Short and descriptive. Prefer 2-5 words after the prefix.

Correct: `feature/user-login`, `fix/null-pointer-on-save`, `sandbox/test-new-orm`
Incorrect: `Feature/UserLogin`, `feature/user_login`, `bugfix/save-error`, `my-branch`, `feature/añadir-módulo!`

Before creating any branch, verify the proposed name against every rule above. If a user asks for a branch name that violates these rules, propose the corrected compliant name instead and explain why.

### Commit Message Format (MANDATORY)

Every commit message MUST begin with one of the following tags, in square brackets, followed by a space and a summary. A commit MAY use more than one tag when it spans several logical changes (see the rules below):

| Tag | When to use it | Example |
|-----|----------------|---------|
| `[ADD]` | Adding a new feature, file, or functionality | `[ADD] Implement user login feature` |
| `[REF]` | Refactoring code without changing its functionality | `[REF] Refactor authentication module` |
| `[DEL]` | Deleting or removing a feature, file, or obsolete code | `[DEL] Remove deprecated API calls` |
| `[WIP]` | Work in progress; incomplete work still under development | `[WIP] Implementing user profile module` |
| `[FIX]` | Fixing bugs or issues | `[FIX] Correct typo in login error message` |
| `[DOC]` | Updating or adding documentation | `[DOC] Update README with setup instructions` |
| `[TEST]` | Adding or updating tests | `[TEST] Add unit tests for user service` |
| `[UPDT]` | Updating existing code | `[UPDT] Updated code for notifications` |

Commit message rules, ALL of which must hold:

1. Every commit MUST start with one of the eight tags above, in square brackets. No commit without a tag. No invented tags such as `[FEAT]`, `[CHORE]`, or `[STYLE]`.
2. A commit MAY carry more than one tag when it genuinely spans several logical changes. Put each tagged change on its own line, and start every one of those lines with its own valid tag. A single-purpose commit is just one tagged line.
3. Each tagged line leads with a concise summary (aim for about 50 characters) that says what that part of the change does, not how. Extra detail is optional and may follow on the same line or on wrapped lines beneath it.
4. Order the tagged lines most significant first. The first line doubles as the commit headline and PR title, so keep it the strongest summary.
5. Still prefer small, atomic commits. But when changes logically ship together, do not split them apart just to keep a single tag; use one tagged line per category instead.

Example of a multi-tag commit:

```
[REF] Group report rows by parent record
[ADD] Clear hover affordance on clickable rows
[FIX] Correct wording on summary figures
[TEST] Cover row grouping and totals
```

Choosing between similar tags:

- New capability that did not exist before: `[ADD]`.
- Behavior was wrong and is now correct: `[FIX]`.
- Behavior unchanged, structure improved: `[REF]`.
- Behavior intentionally changed or modernized but not a new feature and not a bug: `[UPDT]`.
- Only documentation files or comments touched: `[DOC]`.
- Only test files touched: `[TEST]`.

### Pre-Commit Checklist (run mentally before EVERY commit)

1. Am I on a compliant branch (`feature/*`, `fix/*`, or `sandbox/*`)? If on `main` or `development`, STOP and create a branch.
2. Does the commit message start with a valid tag, and is every tag used one from the table above?
3. Does each tagged line lead with a concise (about 50 character) summary?
4. Is the branch name lowercase, hyphenated, and correctly prefixed?

If any answer is no, correct it before committing. Do not proceed and fix later.

### Branch Lifecycle (recommended flow)

1. Sync first: update your local `development` (`git pull`) before creating a new branch, so work starts from the latest integrated code.
2. Branch from `development` for `feature/*` and `sandbox/*` work. `fix/*` branches normally also come from `development`; branch from `main` only for urgent production hotfixes the user explicitly requests.
3. Commit in small, atomic units. A commit should represent a coherent change; if it spans categories, give each logical change its own tagged line.
4. Merge back into `development` via pull request when the work is complete.
5. `development` is merged into `main` only when preparing a production release, and only when the user asks for it.
6. Delete the working branch once it is merged or no longer needed (see Branch Cleanup below).

```
main <------ development <------ feature/*  fix/*  sandbox/*
(production)  (integration)          (working branches)
```

### Branch Cleanup (MANDATORY)

Working branches are disposable. Delete one as soon as it is merged or has outlived its
usefulness, so the branch list only ever shows live work. A branch nobody is going to
finish is noise, and the longer it sits the further it drifts from `development`.

Delete a working branch when any of these is true:

- Its pull request has been merged into `development` (or into `main` for a hotfix).
- The work was abandoned or superseded by another branch.
- A `sandbox/*` experiment has answered its question. These are expected to die young;
  keep the finding, not the branch.
- It has gone stale enough that it would have to be rebuilt on current `development`
  anyway. Re-branch from `development` instead of reviving it.

Rules, ALL of which must hold:

1. **Ask the user before deleting any branch.** Deletion is the default outcome, not the
   default action: propose it, then act on the answer.
2. **NEVER delete `main` or `development`.**
3. **Delete both copies.** A branch is not cleaned up while the remote still lists it.
4. **Confirm it is merged first.** Use `git branch -d` (lowercase), which refuses to
   delete a branch holding unmerged commits. `git branch -D` (force) discards those
   commits, so it requires explicit user approval, and only for work being abandoned.
5. **Prune stale remote-tracking refs** after someone deletes a branch on the remote
   (for example, a PR merged with "delete branch" on GitHub), so `git branch -a` stays
   honest.

```bash
git branch --merged development         # local branches already merged: safe to delete
git branch -d feature/user-login        # local; refuses if it holds unmerged commits
git push origin --delete feature/user-login   # remote
git fetch --prune                       # drop remote-tracking refs for deleted branches
git branch --no-merged development      # still-unmerged branches: candidates to review
```

Review the branch list periodically (`git branch -a`), not only right after a merge.
When a branch is unmerged but stale, report it to the user with what is on it rather
than deleting it, they decide whether it is worth finishing.

### Pull Requests

- PRs from working branches target `development`, not `main`, unless the user explicitly says otherwise (for example, a release or an urgent hotfix promotion).
- PR titles should follow the same tag convention as commit messages (for example, `[ADD] User login feature`).
- Do not open a PR whose latest commits are tagged `[WIP]`. Finish the work, or tell the user it is still in progress, before opening the PR.
- PR descriptions should briefly list what changed and how it was tested.

### Safety Rules

- NEVER force-push (`git push --force` or `--force-with-lease`) to `main` or `development`. Force-pushing a personal working branch requires explicit user approval.
- NEVER rewrite history (rebase, amend, reset) on `main` or `development`.
- Never commit secrets, credentials, `.env` files, or generated artifacts. If one shows up in `git status`, flag it and suggest adding it to `.gitignore` instead of committing it.
- Prefer creating a new commit over amending an existing one that has already been pushed.
- Run the project's tests or build (when they exist) before committing anything other than `[WIP]` or `[DOC]` changes.

## Quick Reference

```
Branch:  feature/short-description | fix/short-description | sandbox/short-description
         (lowercase, hyphens, no special characters)

Commit:  [ADD] | [REF] | [DEL] | [WIP] | [FIX] | [DOC] | [TEST] | [UPDT]
         One tagged line per logical change (multiple tags allowed).
         Each line: tag + concise summary (about 50 chars) + optional detail.

Flow:    branch from development -> commit -> PR into development -> release to main
Clean:   after a merge (or when a branch is abandoned), delete it local + remote
         `git branch -d <name>` + `git push origin --delete <name>` (ask first)
```

## Scope

These rules apply to all git operations performed in this repository: creating branches, committing, merging, and opening pull requests. When a user gives an instruction that conflicts with these guidelines, point out the conflict and ask before deviating.
</content>
