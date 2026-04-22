import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configuration.config import NEO4J_CONFIG
from neo4j import GraphDatabase


WARNING_TEXT = (
    "This will permanently delete all Neo4j nodes, relationships, indexes, and constraints."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reset the Neo4j graph database.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    return parser


def confirm_reset(skip_prompt: bool) -> bool:
    if skip_prompt:
        return True

    print(WARNING_TEXT)
    answer = input("Type 'RESET' to continue: ").strip()
    return answer == "RESET"


def fetch_names(driver, show_cypher: str, key: str, database_: str | None = None) -> list[str]:
    records, _, _ = driver.execute_query(show_cypher, database_=database_)
    return [record[key] for record in records if record.get(key)]


def reset_graph() -> None:
    driver = GraphDatabase.driver(**NEO4J_CONFIG)
    try:
        driver.execute_query("MATCH (n) DETACH DELETE n")

        constraint_names = fetch_names(
            driver,
            "SHOW CONSTRAINTS YIELD name RETURN name",
            "name",
        )
        for name in constraint_names:
            driver.execute_query(f"DROP CONSTRAINT `{name}` IF EXISTS")

        index_names = fetch_names(driver, "SHOW INDEXES YIELD name RETURN name", "name")
        for name in index_names:
            driver.execute_query(f"DROP INDEX `{name}` IF EXISTS")
    finally:
        driver.close()


def main() -> int:
    args = build_parser().parse_args()
    if not confirm_reset(args.yes):
        print("Reset cancelled.")
        return 1

    reset_graph()
    print("Neo4j graph reset completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
