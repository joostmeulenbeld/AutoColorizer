import urllib
import sqlite3
import os
import convertimage
from multiprocessing import Pool
from math import floor
from itertools import accumulate
import errno

class DBWrapper:
    """Wrapper for the sqlite database to add new images or get image file names"""

    def __init__(self, images_folder='images_jpg', dbname='imagedb.db', images_folder_npy='images_npy'):
        """Open connection to the database and create a cursor. Standard image folder: 'images'
        Don't forget to dbwrapper.close_connection() when done using the database
        """

        self.images_folder = images_folder
        self.images_folder_npy = images_folder_npy
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
        """Close the database connection after committing any uncommitted results"""
        self.conn.commit()
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

    def get_jpg_filenames(self, bool_checked=2, bool_good=2, limit=-1, random=True):
        """Returns a list of all relative paths to jpg files
        INPUT:
            The two bools specify whether to get only checked or good files:
                0: no,
                1: yes,
                2: doesn't matter
            limit=-1: set the amount of file names to retrieve, limit=-1 doesn't limit amount
            random=True: order the results randomly or sorted by id

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
        if random:
            query += " ORDER BY RANDOM() "
        if limit > -1:
            query += " LIMIT " + str(limit)
        query += ";"

        self.cursor.execute(query)
        results = [(id, self.path_to_image_jpg(id), checked, good) for (id, checked, good) in self.cursor]
        return results

    def get_flickr_url_params(self, id):
        """Return all parameters required to construct the Flickr URL"""

        self.cursor.execute('''SELECT id, server, secret FROM images WHERE id=?;''', (id,))
        result = self.cursor.fetchone()
        params = {'id': id, 'server': result[1], 'secret': result[2]}
        return params


    def add_image(self, id, server, secret, title, checked=0, good=0, commit=True):
        """Add an image to the database. Defaults to the unchecked state
        IMPORTANT: returns the path under which to save the image!
        """

        checked = 0
        good = 0
        query = '''INSERT INTO images (id, server, secret, title, checked, good) VALUES (?,?,?,?,?,?)'''
        data = (id, server, secret, title, checked, good)
        self.cursor.execute(query, data)
        if commit:
            self.conn.commit()
        return self.path_to_image_jpg(id)

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

    def path_to_image_jpg(self, id):
        """Generate the path to an image in its jpg form based on its id"""
        return os.path.join(self.images_folder, id + '.jpg')

    def path_to_image_npy(self, id):
        """Generate the path to an image in its numpy form based on its id"""
        return os.path.join(self.images_folder_npy, id + '.npy')


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
        idlist_db = [id for (id,) in self.cursor.execute("SELECT id FROM images;")]
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
                path = self.path_to_image_jpg(id)
                print("Remove file: " + path)
                os.remove(path)
            for id in db_not_in_files:
                print("Remove from database: " + id)
            self.cursor.executemany("DELETE FROM images WHERE id=?", [(i,) for i in db_not_in_files])
            self.conn.commit()
        return db_not_in_files

    def clean_database(self):
        """Clean the database and folder;
        WARNING: this may remove images and database entries"""
        self.check_integrity(clean=True)
        convertimage.check_image_dimensions([jpgfilename for (_,jpgfilename,_,_) in self.get_jpg_filenames()], clean=True)
        self.check_integrity(clean=True)

    def create_numpy_files(self):
        """Get all id's from database, check numpy folder for these images, and create missing files
        This is not used at the time; rather only numpy files of complete batches are saved by create_training_sets
        """

        # Get all images in the database that are checked and found good
        idlist_db = set([id for (id,) in self.cursor.execute("SELECT id FROM images;")])

        # Get all files that are already converted (remove the file extension to obtain image ID)
        idlist_files = set([os.path.splitext(filename)[0] for filename in os.listdir(self.images_folder_npy)])

        # Calculate the missing files
        missing_files = idlist_db - idlist_files
        print("Number of numpy files to be created: " + str(len(missing_files)))

        # Convert the missing files
        pool = Pool(processes=os.cpu_count())
        pool.map(self.convert_jpg_image_to_numpy_by_id, missing_files)

    def get_ids(self, bool_checked=2, bool_good=2, limit=-1):
        return [id for (id,_,_,_) in self.get_jpg_filenames(bool_checked=bool_checked, bool_good=bool_good, limit=limit)]

    def create_training_sets(self, imagesize=150, batch_size=50, fractions=(0.5, 0.25, 0.25), foldernames=("training", "validation", "test"), prefix=""):
        """Create training/test/validations sets (can be any amount of sets)
        INPUT:
            batch_size: training batch size
            fractions: tuple containing the fractions of the training/validation/test sets (sum should be 1)
            foldername: tuple containing the folder names of the different sets
        """

        if len(fractions) != len(foldernames):
            print("number of fractions does not match number of folders")
            return

        if sum(fractions) != 1:
            print("sum of fractions is not equal to 1")
            return

        foldernames = [prefix+"_"+foldername for foldername in foldernames]

        # Try to create all folders required
        for folder in foldernames:
            try:
                os.makedirs(folder)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

        self.clean_database()

        # Get list of good image ID's
        good_image_ids = self.get_ids(bool_checked=2, bool_good=2)

        # Get the number of good images
        number_good_images = len(good_image_ids)

        # Calculate number of batches per set
        number_of_batches_per_set = [floor((number_good_images*fraction)/batch_size) for fraction in fractions]
        number_of_batches_per_set_cum = [0]
        number_of_batches_per_set_cum.extend(list(accumulate(number_of_batches_per_set)))
        print(number_of_batches_per_set)
        # Create the batches of image ID's
        batches = [good_image_ids[i:i+batch_size] for i in range(0, number_good_images, batch_size)]
        batches_per_set = [batches[number_of_batches_per_set_cum[i]:number_of_batches_per_set_cum[i+1]] for i in range(len(number_of_batches_per_set_cum)-1)]
        for (i_set, batches_this_set) in enumerate(batches_per_set):
            for (i_batch, batch) in enumerate(batches_this_set):
                print("set " + foldernames[i_set] + ", batch: " + str(i_batch+1) + " of " + str(len(batches_this_set)))
                jpgfilenames = [self.path_to_image_jpg(id) for id in batch]
                npyfilename = os.path.join(foldernames[i_set], prefix + '_batch_' + str(i_batch) + '.npy')
                convertimage.create_batch_and_save(jpgfilenames, npyfilename, imagesize)

    def convert_jpg_image_to_numpy_by_id(self, id):
        """Convert a specific image from jpg to npy format by id"""
        print("Converting image: " + id)
        convertimage.convert_image_to_YCbCr_and_save(self.path_to_image_jpg(id), self.path_to_image_npy(id))





if __name__ == "__main__":
    wrapper = DBWrapper(dbname='landscape.db', images_folder='landscape_jpg')
    wrapper.create_training_sets(imagesize=128, batch_size=1000, fractions=(1.0,), foldernames=('training',), prefix="landscape")

    # wrapper.clean_database()
    # Check integrity of the database:

    #Get paths to files:
    #ids = [id for (id,_,_,_) in wrapper.get_jpg_filenames(bool_checked=2, bool_good=2, limit=-1)]
    #print(len(ids))
    #wrapper.set_checked(ids)
    #print('set_checked done')
    #wrapper.set_good(ids, good=1)
    #print('set_good done')


    #Update certain parts, idlist contains a list of ID's [ID1, ID2 etc]
    #wrapper.set_checked(idlist, checked=1)
