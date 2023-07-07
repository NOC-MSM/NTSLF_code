"""
SFTP methods to handle file transfers
"""

import os
import pysftp
import socket
import logging
import config


class Uploader:
    def __init__(self,
                hostname:str= "livftp.noc.ac.uk",
                username:str= config.username,  # type:string
                password:str= config.password,  # type:string
                #local_dir:str= "/projectsa/surge_archive/figures/surge_anom_latest/",
                local_dir:str = "/Users/jelt/Downloads/surge_anom_latest/",
                remote_dir:str= "/local/users/ntslf/pub/ntslf_surge_animation/"):

        self.hostname = hostname
        self.username = username
        self.password = password
        self.local_dir = local_dir
        self.remote_dir = remote_dir

        fmt = '%(asctime)s | %(levelname)s | %(message)s'
        logging.basicConfig(level=logging.DEBUG,
                            format=fmt,
                            handlers=[logging.StreamHandler(),
                                      logging.FileHandler("sftp.log")])
        logging.info("=========================")
        logging.info("Start new logging process")

        self.process()

    def process(self):

        # First try/except block
        try:
            file_list = os.listdir(self.local_dir)

            # Validating if there are files
            if not file_list:
                logging.info("No files to upload, waiting for new files...")


            with pysftp.Connection(self.hostname, username=self.username, password=self.password) as sftp:
                logging.info("Connection established with FTP server")

                for file_name in file_list:
                    # temporarily chdir to remote_dir
                    with sftp.cd(self.remote_dir):
                        logging.info(f"Accessing {self.remote_dir}")
                        # Second try/except block
                        try:
                            file_path = os.path.join(self.local_dir, file_name)
                            if os.path.isfile(file_path):
                                logging.info(f"Uploading {file_name}...")
                                sftp.put(file_path)
                        except Exception as e:
                            logging.error(f'Error uploading {file_name}: {str(e)}')
                            continue

                logging.info(f"Uploaded all files to {self.remote_dir}")

        except socket.gaierror:
            logging.critical('Invalid Hostname.')

        except pysftp.ConnectionException:
            logging.critical(f'ConnectionException. username: {self.username}, password: {self.password} or '
                             'remote directory path: {self.remote_dir}')
        except pysftp.AuthenticationException:
            logging.critical(f'AuthenticationException. username: {self.username}, password: {self.password} or '
                             'remote directory path: {self.remote_dir}')

        except FileNotFoundError:
            logging.critical(f'{self.local_dir} is not a valid directory.' 
                              'Please check path informed.')

if __name__ == '__main__':

    #tt = Uploader(local_dir="/projectsa/surge_archive/figures/surge_anom_latest/")
    Uploader(local_dir="/Users/jelt/Downloads/surge_anom_latest/",
             remote_dir="/local/users/ntslf/pub/ntslf_surge_animation/")