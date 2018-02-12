import pandas as pd

from StringIO import StringIO
try:
    from pandas.io.common import ZipFile
except ImportError:
    from zipfile import ZipFile

from polara.tools.movielens import get_movielens_data

def get_bx_data(file_path, get_ratings=True, get_users=False, get_books=False):
    ratings = users = books = None
    with ZipFile(file_path) as zfile:
        zip_files = pd.Series(zfile.namelist())
        zip_file = zip_files[zip_files.str.contains('ratings', flags=2)].iat[0]
        
        delimiter = ';'
        if get_ratings:
            zdata = zfile.read(zip_file)
            ratings = pd.read_csv(StringIO(zdata), sep=delimiter, header=0, engine='c')

        if get_users:
            zip_file = zip_files[zip_files.str.contains('users', flags=2)].iat[0]
            with zfile.open(zip_file) as zdata:
                users = pd.read_csv(zdata, sep=delimiter, header=0, engine='c',)

        if get_books:
            zip_file = zip_files[zip_files.str.contains('books', flags=2)].iat[0]
            with zfile.open(zip_file) as zdata:
                books = pd.read_csv(zdata, sep=delimiter, header=0, engine='c',
                                    quoting=1, escapechar='\\',
                                    usecols=['ISBN', 'Book-Author', 'Publisher'])

    res = [data.rename(columns=lambda x: x.lower().replace('book-', '').replace('-id', 'id'))
           for data in [ratings, users, books] if data is not None]
    return res


def get_ml_data(file_path, get_ratings=True, get_genres=True, split_genres=False, meta_path=None, fixes_path=None):
    data, genres = get_movielens_data(file_path, get_ratings=True,
                                      get_genres=True, split_genres=False)
    genres = genres.assign(movienm=lambda x: x.movienm.str.decode('unicode_escape'))
    
    if fixes_path:
        id_fix = pd.read_csv(fixes_path)
        genres.movieid.replace(id_fix.set_index('ml1mid').movieid, inplace=True)
        genres = genres.drop_duplicates(subset='movieid')
        data.movieid.replace(id_fix.set_index('ml1mid').movieid, inplace=True)
        data = data.drop_duplicates()
        
    meta_info = pd.read_csv(meta_path, sep=';', na_filter=False).set_index('movieid')
    meta_cols = meta_info.columns
    meta_info.loc[:, meta_cols] = meta_info.loc[:, meta_cols].applymap(lambda x: x.split(',') if x else [])