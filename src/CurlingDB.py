
from utils.DBConn import DBConn

class CurlingDB(DBConn):

    def __init__(self, db_name: str='curling_db.db', filepath: str = './'):
        super().__init__(db_name, filepath, './sqlite_scripts/')
    
