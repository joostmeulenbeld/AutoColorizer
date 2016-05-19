import flickrapi
from pprint import pprint
import string
import urllib.request
import math
from datetime import date, timedelta
from dbwrapper import DBWrapper
import getapikey
import shutil
import requests

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

    def flickr_url_from_photo_object(self, photo, image_size='q'):
        """Construct the static flickr url with which you can download the photo
        Input:
            photo object
            image_size: see https://www.flickr.com/services/api/misc.urls.html
        Output:
            flickr static url
        """
        # return the url and the id of the photo
        return self.flickr_url({'id': photo.get('id'), 'server': photo.get('server'), 'secret': photo.get('secret')}, image_size = image_size)


    def flickr_url(self, params, image_size='q'):
        """Construct the static flickr url with which you can download the photo
        Input:
            parameter dictionary containing 'id', 'server', and 'secret'
            image_size: see https://www.flickr.com/services/api/misc.urls.html
        Output:
            flickr static url
        """
        # return the url and the id of the photo
        return 'https://farm{farm_id}.staticflickr.com/{server_id}/{id}_{secret}_{imsize}.jpg'.format(farm_id=2, # Default to farm 1
            server_id=params['server'], 
            id=params['id'], 
            secret=params['secret'], 
            imsize=image_size)

    def download_images(self, text='fruit', num_images=20000):
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
        for weeksback in range(0,3000,1):

            if counter > num_images:
                break

            final_week_day -= timedelta(days=7)
            begin_week_day = final_week_day - timedelta(days=6)

            # Start downloading images
            for (counter_in_week, photo) in enumerate(self.flickr.walk(tag_mode='all',
                    text=text,
                    sort='relevance', #relevant photos first, otherwise the result is crap
                    min_taken_date=begin_week_day.isoformat(),
                    max_taken_date=final_week_day.isoformat(),
                    per_page=100)):

                # Download max 100 images from 1 single week to keep the quality high
                if counter_in_week >= 100:
                    print("Downloaded 100 images between " + begin_week_day.isoformat() + " and " + final_week_day.isoformat())
                    break

                if counter > num_images:
                    print("Downloaded " + str(num_images) + " images, now stopping")
                    break


                # Check if the photo ID is already in the database.  If so, go
                # to the previous week
                if self.wrapper.image_in_db(photo.get('id')):
                    print("\tImage already exists: " + photo.get('id') + ", going to previous week")
                    break
                else:
                    counter += 1
                    # Create (0005/1000) progress string
                    progress = ("({:" + str(2 + math.ceil(math.log10(num_images))) + 'd}/' + str(num_images) + ')').format(counter)

                    url = self.flickr_url_from_photo_object(photo)
                    # Add the image to the database (function returns the disk
                    # location where to store the image)
                    filename = self.wrapper.add_image(photo.get('id'), photo.get('server'), photo.get('secret'), self.__printable(photo.get('title')))
                    
                    response = requests.get(url)
                    
                    file = open(filename, 'wb')
                    file.write(response.content)
                    file.close()

                    print(progress + " Image downloaded: " + photo.get('id'))

        #Get the new amount of photos in database and report amount of
        #downloaded images

        number_of_photos_new = self.wrapper.count_images()
        return (number_of_photos_new - number_of_photos_old)

    def download_missing_images(self):
        mising_images = self.wrapper.check_integrity()
        for id in mising_images:
            self.flickr_url(id)



if __name__ == "__main__":
    ff = FruitFetcher()
    #ff.wrapper.clear_database(reallydoit=True)
    print("Number of downloaded images: " + str(ff.download_images()))
    #ff.wrapper.check_integrity(clean=False)
    print(ff.wrapper.count_images(bool_checked=1))
