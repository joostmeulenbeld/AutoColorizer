import urllib
import sqlite3
import os

class DBWrapper:
    """Wrapper for the sqlite database to add new images or get image file names"""

    def __init__(self, images_folder='images', dbname='imagedb.db'):
        """Open connection to the database and create a cursor. Standard image folder: 'images'
        Don't forget to dbwrapper.close_connection() when done using the database
        """

        self.images_folder = images_folder
        self.dbname = dbname
        self.__open_connection()

    def __open_connection(self):
        """Open connection to database if not already opened. This function is already
        called by the constructor of this class.
        """
        # check if the database exists, otherwise create the table
        create_table = not os.path.isfile(self.dbname)

        self.conn = sqlite3.connect(self.dbname)
        self.cursor = self.conn.cursor()
        if create_table:
            print("Database " + self.dbname + " was not found, creating a new database")
            print("Current working directory: " + os.getcwd())
            self.setup_database()

    def clear_database(self, reallydoit=False):
        """Delete all contents from the database, procede with caution!
        This will also setup a clean new database
        """

        if reallydoit:
            print("Now deleting contents of database and re-configuring...")
            self.cursor.execute('''DROP TABLE images''')
            self.conn.commit()
            self.setup_database()
            print("Done!")

        
    def close_connection(self):
        """Close the database connection"""

        self.conn.close()

    def setup_database(self):
        """Call this function if no database has been setup yet"""

        self.cursor.execute('''CREATE TABLE images
            (id text, server text, secret text, title text, checked integer, good integer);''')
        self.conn.commit()
        return self.cursor.fetchall()

    def count_images(self, bool_checked=2, bool_good=2):
        """Returns the number of images in the database
        The two bools: 
            0: no, 
            1: yes, 
            2: doesn't matter
        """
        if bool_checked not in [0,1,2] or bool_good not in [0,1,2]:
            print("Boolean should be in range [0,1,2]")
            return -1

        query = "SELECT Count(*) FROM images"
        if bool_checked in [0,1] or bool_good in [0,1]:
            query += " WHERE "
            if bool_checked in [0,1]:
                query += "checked=" + str(bool_checked)
            if bool_checked in [0,1] and bool_good in [0,1]:
                query += " AND "
            if bool_good in [0,1]:
                query += "good=" + str(bool_good)
        query += ";"

        self.cursor.execute(query)
        return self.cursor.fetchone()[0]

    def get_jpg_filenames(self, bool_checked=2, bool_good=2, limit=-1):
        """Returns a list of all relative paths to jpg files
        INPUT:
            The two bools specify whether to get only checked or good files: 
                0: no, 
                1: yes, 
                2: doesn't matter
            limit=-1: set the amount of file names to retrieve, limit=-1 doesn't limit amount

        OUTPUT:
        List of tuples: [(id, path/to.jpg, checked, good), ...]
        """

        query = '''SELECT id, checked, good FROM images'''

        if bool_checked in [0,1] or bool_good in [0,1]:
            query += " WHERE "
            if bool_checked in [0,1]:
                query += "checked=" + str(bool_checked)
            if bool_checked in [0,1] and bool_good in [0,1]:
                query += " AND "
            if bool_good in [0,1]:
                query += "good=" + str(bool_good)
        if limit > -1:
            query += " LIMIT " + str(limit)
        query += ";"

        self.cursor.execute(query)
        results = [(id, self.path_to_image(id), checked, good) for (id, checked, good) in self.cursor]
        return results

    def get_flickr_url_params(self, id):
        """Return all parameters required to construct the Flickr URL"""

        self.cursor.execute('''SELECT id, server, secret FROM images WHERE id=?;''', (id,))
        result = self.cursor.fetchone()
        params = {'id': id, 'server': result[1], 'secret': result[2]}
        return params


    def add_image(self, id, server, secret, title, checked=0, good=0):
        """Add an image to the database. Defaults to the unchecked state
        IMPORTANT: returns the path under which to save the image!
        """

        checked = 0
        good = 0
        query = '''INSERT INTO images (id, server, secret, title, checked, good) VALUES (?,?,?,?,?,?)'''
        data = (id, server, secret, title, checked, good)
        self.cursor.execute(query, data)
        self.conn.commit()
        return self.path_to_image(id)

    def image_in_db(self, id):
        """Return 1 if an image ID is already in the database, 0 if it is not"""
        self.cursor.execute('''SELECT Count(*) FROM images WHERE id=?''', (id,))
        return self.cursor.fetchone()[0] != 0

    def set_checked(self, idlist, checked=1):
        """Set all images by ID checked, 1=checked, 0=not checked"""
        if checked not in [0,1]:
            print("checked value should be in [0,1]")
            return

        query = "UPDATE images SET checked={} WHERE id=?".format(str(checked))
        idlist = [(id,) for id in idlist]
        self.cursor.executemany(query, idlist)
        self.conn.commit()

    def path_to_image(self, id, extension='jpg'):
        """Generate the path to an image based on its id, defaulting to .jpg"""
        return os.path.join(self.images_folder, id + '.' + extension)

    def set_good(self, idlist, good=0):
        """Set all images by ID good, 1=good, 0=not good"""
        if good not in [0,1]:
            print("checked value should be in [0,1]")
            return

        query = "UPDATE images SET good={} WHERE id=?".format(str(good))
        idlist = [(id,) for id in idlist]
        self.cursor.executemany(query, idlist)
        self.conn.commit()

    def check_integrity(self, clean=False):
        """Check if all image files are in the database and all database entries have a file
        Set clean=True if you want to remove entries from database and files from directory if they don't match
        
        OUTPUT: list of ID's of database entries that don't have an associated file
        """
        #prepare the lists of files and database entries
        idlist_db = [id for (id,) in self.cursor.execute("SELECT id FROM images")]
        idlist_files = [os.path.splitext(filename)[0] for filename in os.listdir(self.images_folder)]
        #compare them and get exclusive ones
        db_not_in_files = set(idlist_db)-set(idlist_files)
        files_not_in_db = set(idlist_files)-set(idlist_db)
        #report
        print("Number of files not in databse: " + str(len(files_not_in_db)))
        print("Number of entries without file: " + str(len(db_not_in_files)))
        #clean entries from database and files
        if clean:
            for id in files_not_in_db:
                path = self.path_to_image(id)
                print("Remove file: " + path)
                os.remove(path)
            for id in db_not_in_files:
                print("Remove from database: " + id)
            self.cursor.executemany("DELETE FROM images WHERE id=?", [(i,) for i in db_not_in_files])
            self.conn.commit()
        return db_not_in_files



if __name__ == "__main__":
    wrapper = DBWrapper(dbname='imagedb.db')
    
    # Check integrity of the database:
    wrapper.check_integrity(clean=True)

    #Get paths to files:
    #print(wrapper.get_jpg_filenames(bool_checked=0, limit=4))

    #Update certain parts, idlist contains a list of ID's [ID1, ID2 etc]
    #wrapper.set_checked(idlist, checked=1)