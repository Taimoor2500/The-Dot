"""Neo4j driver wrapper (works against AuraDB Free or local Docker Neo4j)."""

import os
from contextlib import contextmanager

from neo4j import Driver, GraphDatabase


def get_driver() -> Driver:
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    return GraphDatabase.driver(uri, auth=(user, password))


@contextmanager
def session():
    driver = get_driver()
    try:
        with driver.session() as s:
            yield s
    finally:
        driver.close()
