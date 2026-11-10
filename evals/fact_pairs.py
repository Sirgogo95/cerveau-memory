"""Calibrating fact deduplication: how similar are two facts that are the same, an
update, a contradiction, distinct, or unrelated, for an embedding model?

    python -m evals.fact_pairs [--embedding NAME]
    python -m evals.fact_pairs --judge     # and how often the model judges pairs right

Deduplication (brain/facts.py) uses two thresholds from the embedding profile:
above `fact_same_score`, a new fact is taken as the same as an old one without asking
the model; between `fact_related_score` and that, the model is asked (same, update,
contradiction, distinct); below, it's new. So `fact_same_score` must sit above every
update, contradiction, and distinct pair (never merge a change silently), and
`fact_related_score` below nearly every same, update, and contradiction pair (never
miss a change). Facts are written in the conversation's language, so pairs mix
English, Spanish, and French.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

PAIRS: dict[str, list[tuple[str, str]]] = {
    "same": [
        ("The user's sister Lucía lives in Montreal.", "Lucía, the user's sister, lives in Montreal."),
        ("The user's favorite café is La Cafetera on Calle El Conde.", "La Cafetera, on Calle El Conde, is the user's favorite café."),
        ("The user is allergic to penicillin.", "The user has a penicillin allergy."),
        ("The user works as a Django developer.", "The user is a developer who works with Django."),
        ("El usuario tiene una perra llamada Canela.", "La perra del usuario se llama Canela."),
        ("El hermano del usuario, Pedro, se casa el 14 de febrero.", "Pedro, hermano del usuario, se casa el 14 de febrero."),
        ("L'utilisateur prend des cours de piano le mardi soir.", "Le mardi soir, l'utilisateur suit des cours de piano."),
        ("The user's sister Lucía lives in Montreal.", "La hermana del usuario, Lucía, vive en Montreal."),
        ("The user stopped eating red meat for their cholesterol.", "El usuario dejó de comer carne roja por el colesterol."),
        ("The user's manager is Priya.", "Priya is the user's manager."),
        # Found live 2026-10-17: rewordings the model wrote for the same fact.
        ("El usuario es padrino del hermano Pedro.", "El usuario es padrino del matrimonio de su hermano Pedro."),
        ("El usuario planea contar una anécdota sobre perderse en el Pico Duarte de niños en su discurso.",
         "El usuario planea contar una anécdota sobre su pérdida infantil en el Pico Duarte en su discurso."),
    ],
    "update": [
        ("The user's sister Lucía lives in Montreal.", "The user's sister Lucía lives in Toronto."),
        ("The customer portal launches on November 14.", "The customer portal launches on December 5."),
        ("The user's guitar lesson is on Tuesday.", "The user's guitar lesson is on Thursday."),
        ("The user rents an apartment in Naco.", "The user rents an apartment in Bella Vista."),
        ("El usuario pesa 82 kilos.", "El usuario pesa 78 kilos."),
        ("La cita con el dentista es el 28 de agosto.", "La cita con el dentista es el 4 de septiembre."),
        ("L'utilisateur travaille chez Banco Popular.", "L'utilisateur travaille chez Banreservas."),
        ("The nightly reports are sent with Celery beat.", "The nightly reports are sent with cron."),
        ("The user's favorite café is La Cafetera.", "The user's favorite café is Café Malecón."),
        ("The user's car is a 2015 Toyota Corolla.", "The user's car is a 2022 Honda Civic."),
        # Worded loosely, as the model really writes them (found live 2026-10-17).
        ("Pedro se casa el 14 de febrero en Puerto Plata.", "La boda de Pedro, el hermano del usuario, es el 3 de marzo."),
        ("The user's sister Lucía lives in Montreal.", "Lucía moved to Toronto for a new hospital job."),
        ("The user works at Banco Popular as a developer.", "The user now works at a startup called Kiwi."),
        ("L'utilisateur prend des cours de piano le mardi.", "Les cours de piano de l'utilisateur sont maintenant le jeudi soir."),
        ("The customer portal launches on November 14.", "Priya moved the portal launch to December because of the security audit."),
        ("Pedro se casa en Puerto Plata el 14 de febrero.", "El usuario confirmó que la boda de su hermano Pedro se movió al 3 de marzo."),
    ],
    "contradiction": [
        ("The user drinks coffee every morning.", "The user doesn't drink coffee."),
        ("The user has a dog.", "The user has no pets."),
        ("El usuario está casado.", "El usuario está soltero."),
        ("The user likes spicy food.", "The user can't stand spicy food."),
        ("L'utilisateur parle allemand.", "L'utilisateur ne parle pas allemand."),
        ("The user's mother knows about the party.", "The user's mother doesn't know about the party."),
        ("The user plans to move to Spain.", "The user decided not to move to Spain."),
        ("The user uses Postgres for Cerveau.", "The user doesn't use Postgres for Cerveau."),
    ],
    "distinct": [
        ("The user's sister Lucía lives in Montreal.", "The user's sister Lucía is a nurse."),
        ("The user's dog is named Canela.", "The user's dog is afraid of thunder."),
        ("El hermano del usuario se llama Pedro.", "El hermano del usuario vive en Santiago."),
        ("The user's favorite café is La Cafetera.", "The user goes to La Cafetera every Saturday."),
        ("The user works as a Django developer.", "The user's manager is Priya."),
        ("L'utilisateur prend des cours de piano.", "L'utilisateur veut jouer une valse de Chopin."),
        ("The user is allergic to penicillin.", "The user takes metformin."),
        ("The customer portal launches on December 5.", "Tomas owns the Stripe integration."),
        ("The user's mother retires in June.", "The user's mother wants to visit Seville."),
        ("El usuario paga las clases de guitarra de su sobrino.", "El sobrino del usuario tiene 12 años."),
        ("El usuario quiere dar un discurso gracioso en la boda de Pedro.",
         "El usuario planea contar una anécdota del Pico Duarte en su discurso."),
    ],
    "unrelated": [
        ("The user's sister Lucía lives in Montreal.", "The billing job uses a Postgres sequence."),
        ("The user's dog is named Canela.", "The user's manager is Priya."),
        ("El usuario pesa 82 kilos.", "L'utilisateur prend des cours de piano."),
        ("The user drinks coffee every morning.", "The customer portal launches on December 5."),
        ("The user's car is a Honda Civic.", "The user's mother retires in June."),
        ("La perra del usuario se llama Canela.", "The user works as a Django developer."),
    ],
}


def similarities(embedder) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for label, pairs in PAIRS.items():
        texts = [t for pair in pairs for t in pair]
        vectors = embedder.embed(texts)
        out[label] = [
            sum(x * y for x, y in zip(vectors[2 * i].vector, vectors[2 * i + 1].vector))
            for i in range(len(pairs))
        ]
    return out


def judge_accuracy(llm) -> tuple[int, int, list[tuple[str, str, str]]]:
    """Run the fact judge on every labelled pair (unrelated ones aside). Update and
    contradiction count as one answer: both replace the old fact. Returns (right,
    total, mistakes as (expected, got, new statement))."""
    from brain.facts import judge
    from brain.types import Fact

    replaces = {"update": "change", "contradiction": "change"}
    right, mistakes = 0, []
    cases = [(a, b, label) for label in ("same", "update", "contradiction", "distinct")
             for a, b in PAIRS[label]]
    for old, new, label in cases:
        got = judge(Fact(0, old), new, llm)
        if replaces.get(got, got) == replaces.get(label, label):
            right += 1
        else:
            mistakes.append((label, got, new))
    return right, len(cases), mistakes


def main(argv: list[str] | None = None) -> int:
    from adapters.embeddings.factory import embedder_from_profile
    from brain.config import load_embedding_profile

    parser = argparse.ArgumentParser(prog="python -m evals.fact_pairs")
    parser.add_argument("--embedding", default="bge-m3")
    parser.add_argument("--judge", action="store_true",
                        help="also run the fact judge (background model) on every pair")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    profile = load_embedding_profile(root / "configs" / "embeddings" / f"{args.embedding}.toml")
    scores = similarities(embedder_from_profile(profile))
    print(f"{profile.model}: cosine similarity per kind of pair\n")
    for label, values in scores.items():
        print(f"  {label:13} min {min(values):.3f}  median {statistics.median(values):.3f}  "
              f"max {max(values):.3f}   ({len(values)} pairs)")
    changes = scores["update"] + scores["contradiction"] + scores["distinct"]
    print(f"\n  same pairs above every change: {sum(s > max(changes) for s in scores['same'])}"
          f"/{len(scores['same'])}  (highest change {max(changes):.3f})")
    worth_asking = scores["same"] + scores["update"] + scores["contradiction"]
    print(f"  lowest same/update/contradiction: {min(worth_asking):.3f}; "
          f"highest unrelated: {max(scores['unrelated']):.3f}")
    print(f"\n  current: fact_same_score {profile.fact_same_score}, "
          f"fact_related_score {profile.fact_related_score}")
    if args.judge:
        from adapters.llm.factory import provider_from_profile
        from brain.config import load_model_profile

        model = load_model_profile(root / "configs" / "models" / "qwen3.5-4b-background.toml")
        right, total, mistakes = judge_accuracy(provider_from_profile(model))
        print(f"\nJudge ({model.name}): {right}/{total} right (update and contradiction count as one)")
        for expected, got, new in mistakes:
            print(f"  expected {expected}, got {got}: {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
