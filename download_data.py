import os
import boto3
import argparse

'''
To download data, use command: python download_data.py in the main project directory

'''


def make_dir_if_not_exists(dir_name):   
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def progressBar(key, current, total, barLength=20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %% | uploading %s' % (arrow, spaces, percent, os.path.split(key)[1]), end='\x1b\r')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1", 'y')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--overwrite', type=str2bool, help='if True, will overwrite local with download from S3',
                      default='False')
  args = parser.parse_args()

  s3 = boto3.resource('s3')
  b = s3.Bucket('photodraw-public')
  keys = [obj.key for obj in b.objects.all()]  
  print('Downloading from photodraw-public')

  # create the directories needed if they do not exist
  new_dirs = list(set([os.path.split(key)[0] for key in keys]))
  [make_dir_if_not_exists(new_dir) for new_dir in new_dirs]

  print('Initiating download from S3 ...')
  for i, s3_object in enumerate(b.objects.all()):
    if (args.overwrite == True or os.path.exists(s3_object.key)==False) and (os.path.splitext(s3_object.key)[1] != '.zip'):
      progressBar(s3_object.key, i+1, len(keys))
      b.download_file(s3_object.key, s3_object.key)
  print()
  print('All done!')
  print()
