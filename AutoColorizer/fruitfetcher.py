import flickrapi
from pprint import pprint
import string
import urllib.request
import math
from datetime import date, timedelta
from DBWrapper import DBWrapper
import getapikey

class FruitFetcher:
    """Download images from flickr, save them to jpg files and save them to the databse"""

    def __init__(self):
        """Initialise fruitfetcher with the given api key and api secret"""
        self.wrapper = DBWrapper(dbname='imagedb.db')

        self.api_key, self.api_secret = getapikey.getkey()
        self.flickr = flickrapi.FlickrAPI(self.api_key, self.api_secret, format='etree')


    def __printable(self, s):
        """Remove all non-printable characters from the array (some photo titles have
        strange characters)
        """

        return ''.join(filter(lambda x: x in string.printable, s))

    def flickr_url(self, photo, image_size='m'):
        """Construct the static flickr url with which you can download the photo
        Input:
            photo object
            image_size: see https://www.flickr.com/services/api/misc.urls.html
        Output:
            flickr static url
        """
        # return the url and the id of the photo
        return 'https://farm{farm_id}.staticflickr.com/{server_id}/{id}_{secret}_{imsize}.jpg'.format(farm_id=1, # Default to farm 1
            server_id=photo.get('server'), 
            id=photo.get('id'), 
            secret=photo.get('secret'), 
            imsize=image_size)


    def download_images(self, text='fruit', num_images=100000):
        """Download images with the given text, sorted on relevance
        to make sure best images are downloaded first
        for every image, first the database is checked if the image is not already there.
        Then it is added to the database, then it is saved to storage.
        INPUT:
            text='fruit': text to search for
            num_images=100000: amoung of images to download
        OUTPUT:
            number of photos added
        """

        #Get the current amount of photos in the database to calculate the
        #amount of downloaded images afterwards
        counter = 0
        number_of_photos_old = self.wrapper.count_images()

        final_week_day = date.today()
        #loop over amount of weeks to go back in time (10 years)
        for weeksback in range(0,520,1):

            if counter > num_images:
                break

            final_week_day -= timedelta(days=7)
            begin_week_day = final_week_day - timedelta(days=6)


            for photo in self.flickr.walk(tag_mode='all',
                    text=text,
                    sort='relevance', #relevant photos first, otherwise the result is crap
                    min_taken_date=begin_week_day.isoformat(),
                    max_taken_date=final_week_day.isoformat(),
                    per_page=100):

                if counter > num_images:
                    break

                # Check if the photo ID is already in the database.  If so, go
                # to the previous week
                if self.wrapper.image_in_db(photo.get('id')):
                    print("\tImage already exists: " + photo.get('id') + ", going to previous week")
                    break
                else:
                    counter += 1
                    progress = ("({:" + str(2 + math.ceil(math.log10(num_images))) + 'd}/' + str(num_images) + ')').format(counter)

                    url = self.flickr_url(photo)
                    # Add the image to the database (function returns the disk
                    # location where to store the image)
                    filename = self.wrapper.add_image(photo.get('id'), photo.get('title'), url)
                    urllib.request.urlretrieve(url, filename)  #Download the file to the local folder
                
                    print(progress + " Image downloaded: " + photo.get('id'))

        #Get the new amount of photos in database and report amount of
        #downloaded images
        number_of_photos_new = self.wrapper.count_images()
        return (number_of_photos_new - number_of_photos_old)



if __name__ == "__main__":
    ff = FruitFetcher()
    #ff.wrapper.clear_database(reallydoit=True)
    print("Number of downloaded images: " + str(ff.download_images()))
    #ff.wrapper.check_integrity(clean=False)
    print(ff.wrapper.count_images(bool_checked=1))
