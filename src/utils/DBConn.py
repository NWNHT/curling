import logging
from os.path import isfile, join
import sqlite3
import time
from typing import List, Optional

logger = logging.getLogger('__main__.' + __name__)

class DBConn:
	instance = None

	def __new__(cls, *args, **kwargs):
		if cls.instance == None:
			cls.instance = super().__new__(cls)
		return cls.instance
	
	def __init__(self, db_name: str, filepath: str='./', script_filepath: str='./SQLite_scripts/'):
		self.data_path = filepath
		self.name = db_name
		self.script_filepath = script_filepath
		self.conn = self.connect()
		self.cursor = self.conn.cursor()
	
	def connect(self):
		"""
		Check if database exists and return cursor, if no database then create one and initialize with script.
		"""
		
		# Create db and make tables if it does not exist
		if not isfile(self.data_path + self.name):
			try:
				self.conn = sqlite3.connect(self.data_path + self.name)
				self.cursor = self.conn.cursor()
				logger.info('No existing database, creating database.')
				self.create_tables()
				return self.conn
			except Exception as e:
				logger.critical(f"Error creating database: {e}")
				quit()
		else:
			try:
				logger.info("Connecting to database.")
				return sqlite3.connect(self.data_path + self.name)
			except sqlite3.Error as e:
				logger.critical("Error connecting to database.")
				quit()
	
	def __del__(self):
		self.cursor.close()
		self.conn.close()
	
	def commit(self):
		"""
		Perform commit on database
		"""

		logger.debug("Committing to database.")
		self.conn.commit()

	def drop_tables(self):
		"""
		Drop all database tables
		"""

		with open(join(self.script_filepath, 'drop_tables.sql'), 'r') as fh:
			commands = fh.read()

		logger.info("Dropping all tables.")
		self.cursor.executescript(commands)
		self.commit()

	def create_tables(self):
		"""
		Create all database tables
		"""

		with open(join(self.script_filepath, 'create_tables.sql'), 'r') as fh:
			commands = fh.read()
		
		logger.info("Creating all tables.")
		self.cursor.executescript(commands)
		self.commit()

	def execute_command(self, command: str, arguments: Optional[tuple], commit: bool=True):
		"""
		Execute arbitrary command
		"""

		logger.debug(f"Executing command {command}.")
		if arguments is None:
			self.cursor.execute(command)
		else:
			self.cursor.execute(command, arguments)
		if commit: self.commit()


	def execute_many(self, command: str, arguments: List[tuple], commit: bool=True):
		"""
		Execute many arbitrary commands
		"""

		logger.debug(f"Executing command {command}.")

		self.cursor.executemany(command, arguments)

		if commit: self.commit()

	def execute_query(self, query: str, arguments: Optional[tuple]=None):
		"""
		Execute arbitrary query
		"""

		logger.debug(f"Executing query {query}.")
		if arguments is None:
			return self.cursor.execute(query)
		else:
			return self.cursor.execute(query, arguments)


if __name__ == '__main__':

	start = time.perf_counter()

	db = DBConn('health_data.db')

	del(db)

	print(f"Total time: {time.perf_counter() - start}")
