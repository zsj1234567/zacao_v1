import yaml
import psycopg2
import os

def load_db_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/db_config.yaml')
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['postgresql']

def get_db_connection():
    config = load_db_config()
    conn = psycopg2.connect(
        host=config['host'],
        port=config['port'],
        user=config['user'],
        password=config['password'],
        database=config['database']
    )
    return conn 