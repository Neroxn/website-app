from os import remove
from werkzeug.security import check_password_hash, generate_password_hash
from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from .db import get_db
from utils import save_temp_dataframe,remove_temp_files
import pandas as pd

bp = Blueprint('auth', __name__)

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'
        elif db.execute(
            'SELECT id FROM user WHERE username = ?', (username,)
        ).fetchone() is not None:
            error = 'User {} is already registered.'.format(username)

        if error is None:
            db.execute(
                'INSERT INTO user (username, password) VALUES (?, ?)',
                (username, generate_password_hash(password))
            )
            db.commit()
            return redirect(url_for('auth.login'))

        flash(error)

    return render_template('register.html')
    
@bp.route('/login', methods=['GET', 'POST'])
def login():
    if session.get("user_id"): #if already login
        return redirect(url_for("workspace"))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None
        user = db.execute(
            'SELECT * FROM user WHERE username = ?', (username,)
        ).fetchone()

        if user is None:
            error = 'Incorrect username.'
        elif not check_password_hash(user['password'], password):
            error = 'Incorrect password.'

        if error is None:
            session.clear()
            session['user_id'] = user['id']
            save_temp_dataframe(pd.DataFrame(),session.get("user_id")) #save empty dataframe after login
            return redirect(url_for('workspace'))

        flash(error)

    return render_template('login.html')
        
@bp.before_app_request
def load_logged_in_user():
    """
    Function triggered before each request automatically
    """
    
    #Get user session
    user_id = session.get('user_id')
    print("User id of the current user is : ",user_id)
    #If user not exist in session, redirect to login page
    if user_id is None and request.endpoint not in ['auth.login','static','auth.register','auth.logout']:
        flash("Please login to continue")
        return redirect(url_for('auth.login'))

        
@bp.route('/logout')
def logout():
    print("Logoutted.")
    remove_temp_files(session.get("user_id"))
    remove_temp_files(session.get("user_id"),head = "models/")
    session.clear()
    return redirect(url_for('auth.login'))