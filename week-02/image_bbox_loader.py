import glob, os
import pandas as pd

from pdb import set_trace as bp


def load_image_bbox_label_list(ROOT_DIR, stage_num, test_only_part):
  train_dicom_dir = os.path.join(ROOT_DIR, 'stage_' + stage_num + '_train_images')
  test_dicom_dir = os.path.join(ROOT_DIR, 'stage_' + stage_num + '_test_images')
  bbox_path = os.path.join(ROOT_DIR, 'stage_' + stage_num + '_train_labels.csv')

  # make DataFrame with images
  det_class_df = pd.read_csv(os.path.join(ROOT_DIR, 'stage_' + stage_num + '_detailed_class_info.csv'))
  bbox_df = pd.read_csv(os.path.join(ROOT_DIR, 'stage_' + stage_num + '_train_labels.csv'))
  comb_bbox_df = pd.concat([bbox_df,
                            det_class_df.drop('patientId',1)], 1)

  comb_bbox_df['bbox'] = comb_bbox_df[['x', 'y', 'width', 'height']].values.tolist()
  comb_bbox_df_comp = comb_bbox_df.drop(['x', 'y', 'width', 'height'], axis=1)
  comb_bbox_df_comp_group = \
    comb_bbox_df_comp.groupby('patientId').agg(
      {'Target':lambda x: list(x), 'class':lambda x: list(x), 'bbox':lambda x: list(x)})

  image_df = pd.DataFrame({'path': glob.glob(os.path.join(train_dicom_dir, '*.dcm'))})
  image_df['patientId'] = image_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])

  img_pat_ids = set(image_df['patientId'].values.tolist())
  box_pat_ids = set(comb_bbox_df['patientId'].values.tolist())
  # check to make sure there is no funny business
  assert img_pat_ids.union(box_pat_ids)==img_pat_ids, "Patient IDs should be the same"

  # get data statistics
  DCM_TAG_LIST = ['PatientAge', 'BodyPartExamined', 'ViewPosition', 'PatientSex']
  
  # merge into one DataFrame
  image_bbox_df = pd.merge(comb_bbox_df_comp_group,
                           image_df,
                           on='patientId',
                           how='left')
  if test_only_part:
    image_bbox_df = image_bbox_df.iloc[:1000]
  
  print(image_bbox_df.shape[0], 'images found')
   
  image_bbox_label_list = []
  for idx, row in image_bbox_df.iterrows():
    if row['Target'][0] == 1:
      image_bbox_label_list.append([row['path'], row['bbox'], row['Target']])

  return image_bbox_label_list
