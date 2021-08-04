import sqlite3

import click
from flask import current_app, g
from flask.cli import with_appcontext


def get_db():
    """
    Get the database.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    """
    Close the database.
    """
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_db():
    """
    Initialize the database as "schemla.sql"
    """
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))


@click.command('init-db')
@with_appcontext
def init_db_command():
    """
    Clear the existing data and create new tables. Register the command init-db so that
    flask init-db will trigger this function.
    """
    init_db()
    click.echo('Initialized the database.')
    
def init_app(app):
    """
    Initialize the app
    """
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
