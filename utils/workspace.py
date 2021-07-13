from .db import *
from flask import flash
import pandas as pd
import numpy as np
import os
import glob

def get_workspaces(user_id):
    """
    Get workspaces of the user
    """
    db = get_db()
    
    #Check user exists
    """
    if db.execute('SELECT username FROM user WHERE id = ?', (user_id,)).fetchone() is None:
        flash('No user with given id, cannot get workspaces')
        return
    """
    #Get workspace_id from db
    query = db.execute('SELECT workspace_id FROM transactions WHERE user_id = ? ORDER BY workspace_id ASC', (user_id,)).fetchall()
    
    #Return workspace_ids
    if query is None:
        return []
    else:
        return np.unique([item for tup in query for item in tup])

def get_workspace(user_id, workspace_id):
    """
    Get the workspace checkpoints list
    """
    db = get_db()
    #Check user exists
    """
    if db.execute("SELECT username FROM user WHERE id = ? AND workspace_id = ?", (user_id, workspace_id)).fetchone() is None:
        flash('No user with given id, cannot get the workspace')
        return
    """
    #Get checkpoint list from db
    query = db.execute('SELECT target_filename FROM transactions WHERE user_id = ?  AND workspace_id = ? ORDER BY target_filename ASC', (user_id,workspace_id)).fetchall()
    
    if query is None:
        return []
    else:
        return [(((item.split('_'))[-1]).split('.'))[0] for tup in query for item in tup]

def create_workspace(user_id, df):
    """
    Create a workspace returns Nothing
    """
    db = get_db()
    loc = None
    

    #Check user exists
    #if db.execute('SELECT username FROM user WHERE user_id = ?', (user_id,)).fetchone() is None:
    #    flash('No user with given id, cannot create workspace')
    #    return
    
    #Get workspace_id from db
    query = db.execute('SELECT workspace_id FROM transactions WHERE user_id = ? ORDER BY workspace_id DESC', (user_id,)).fetchone()
    
    #If there is a workspace then add new workspace else create first workspace
    if query is None:
        db.execute(
                'INSERT INTO transactions (user_id, workspace_id, source_filename, target_filename, description) VALUES (?, ?, ?, ?, ?)',
                (user_id, 0, (str(user_id) + '_0_0.csv'), (str(user_id) + '_0_0.csv'), 'INITIALIZE')
            )
        loc = (str(user_id) + '_0_0.csv')
    else:
        last_workspace_id = query[0]
        db.execute(
                'INSERT INTO transactions (user_id, workspace_id, source_filename, target_filename, description) VALUES (?, ?, ?, ?, ?)',
                (user_id, last_workspace_id+1, (str(user_id) + '_'+str(last_workspace_id+1)+'_0000.csv'), (str(user_id) + '_'+str(last_workspace_id+1)+'_0.csv'), 'INITIALIZE')
            )
        loc = (str(user_id) + '_'+str(last_workspace_id+1)+'_0.csv')
        
    
    db.commit()
    df.to_csv(('./csv/' + loc), index=False)
    return

def delete_workspace(user_id, workspace_id):
    """
    Delete a workspace returns Nothing
    """
    db = get_db()
    
    #Check user exists

    """
    if db.execute('SELECT username FROM user WHERE id = ?', (user_id,)).fetchone() is None:
        flash('No user with given id, cannot create workspace')
        return
        """

    #Check workspace exists
    query = db.execute('SELECT target_filename FROM transactions WHERE user_id = ? AND workspace_id = ?', (user_id, workspace_id)).fetchall()
    if  query is None:
        flash('No workspace for given workspace_id, cannot delete the workspace')
        return
    
    #removes csv files
    files = glob.glob('./csv/'+ str(user_id) + '_' + str(workspace_id) + '_*.csv')
    for f in files:
        print(f)
        os.remove(f)
    
    #remove transactions
    db.execute('DELETE FROM transactions WHERE user_id = ? AND workspace_id = ?', (user_id, workspace_id))
    db.commit()
    
    return
    
def get_checkpoint(user_id, workspace_id, checkpoint_id):
    """
    Get checkpoint
    """
    db = get_db()
    """
    #Check user exists
    if db.execute('SELECT username FROM user WHERE id = ?', (user_id,)).fetchone() is None:
        flash('No user with given id, cannot get checkpoint')
        return
        """

    #Check workspace exists
    loc = str(user_id) + '_' + str(workspace_id) + '_' + str(checkpoint_id) + '.csv'
    query = db.execute('SELECT target_filename FROM transactions WHERE user_id = ? AND workspace_id = ? AND target_filename = ? ORDER BY target_filename DESC', (user_id, workspace_id, loc)).fetchone()
    if  query is None:
        flash('No workspace for given checkpoint_id, cannot get checkpoint')
        return
    
    #Return df if exists
    df = pd.read_csv('./csv/'+ query[0])
    return df

def add_checkpoint(user_id, workspace_id, source_id, df, desc):
    """
    Add Dataframe to given workspace returns df
    """
    
    db = get_db()
    #Check user exists
    """
    if db.execute('SELECT username FROM user WHERE id = ?', (user_id,)).fetchone() is None:
        flash('No user with given id, cannot add checkpoint')
        return
        """

    #Check workspace exists
    query = db.execute('SELECT target_filename FROM transactions WHERE user_id = ? AND workspace_id = ? ORDER BY target_filename DESC', (user_id, workspace_id)).fetchone()
    if  query is None:
        flash('No workspace for given workspace_id, cannot add checkpoint')
        return
        
    #Get last checkpoint 
    last  = query[0]
    name = last.split('_')
    num = int(name[-1].split('.')[0])
    target = (str(user_id) + '_' + str(workspace_id) + '_' + str(num+1) + '.csv')
    source = (str(user_id) + '_' + str(workspace_id) + '_' + str(source_id) + '.csv')
    
    #Add checkpoint
    db.execute('INSERT INTO transactions (user_id, workspace_id, source_filename, target_filename, description) VALUES (?, ?, ?, ?, ?)', (user_id, workspace_id, source, target, desc))
    db.commit()
    
    #Save dataframe
    df.to_csv(('./csv/' + target), index=False)
    
    return df

def delete_checkpoint(user_id, workspace_id, checkpoint_id):
    """
    Delete last checkpoint and if exists return one before, if not return None
    """
    
    db = get_db()
    """
    #Check user exists
    if db.execute('SELECT username FROM user WHERE id = ?', (user_id,)).fetchone() is None:
        flash('No user with given id, cannot delete checkpoint')
        return None
        """
    #Check workspace exists
    loc = str(user_id) + '_' + str(workspace_id) + '_' + str(checkpoint_id) + '.csv'
    query = db.execute('SELECT target_filename FROM transactions WHERE user_id = ? AND workspace_id = ? AND target_filename = ? ORDER BY target_filename DESC', (user_id, workspace_id, loc)).fetchone()
    if  query is None:
        flash('No workspace for given checkpoint_id, cannot delete checkpoint')
        return None
    
    #Delete csv file
    name = query[0]
    f = glob.glob('./csv/'+ name)[0]
    print(name,f)
    os.remove(f)
    
    #Delete transaction
    db.execute('DELETE FROM transactions WHERE user_id = ? AND workspace_id = ? AND target_filename = ?', (user_id, workspace_id, name))
    db.commit()
    
    last = db.execute('SELECT target_filename FROM transactions WHERE user_id = ? AND workspace_id = ? ORDER BY target_filename DESC', (user_id, workspace_id)).fetchone()
    
    if last is None:
        return None
    else:
        df = pd.read_csv('./csv/' + last[0])
        return df