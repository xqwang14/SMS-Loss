import pickle
import numpy as np
import pandas as pd

#Modify the root to the parent folder to aviod errors when importing avion
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
#sys.path.append('/home/wangxiaoqi/avion/third_party/decord/python')
#-------------------end of modification----------------------------------

from avion.utils.evaluation_ek100mir import get_mAP, get_nDCG

def load_pickle(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def sum_sim_mats(pickle_files):
    """Sum the sim_mat arrays from multiple pickle files."""
    sim_mats = []
    for file_path in pickle_files:
        data = load_pickle(file_path)
        sim_mats.append(data['sim_mat'])
    
    # Ensure that all sim_mat arrays have the same shape
    for i in range(1, len(sim_mats)):
        if sim_mats[i].shape != sim_mats[0].shape:
            raise ValueError("All sim_mat arrays must have the same shape.")
    
    # Sum the sim_mat arrays element-wise
    summed_sim_mat = np.sum(sim_mats, axis=0) / len(sim_mats)
    
    return summed_sim_mat

def rank_aggregation(pickle_files):
    sim_mats = []
    for file_path in pickle_files:
        data = load_pickle(file_path)
        sim_mats.append(data['sim_mat'])
    """Aggregate ranks from multiple similarity matrices."""
    num_queries, num_items = sim_mats[0].shape
    aggregated_ranks = np.zeros((num_queries, num_items))
    
    for sim_mat in sim_mats:
        ranks = np.argsort(np.argsort(sim_mat, axis=1), axis=1)
        aggregated_ranks += ranks
    
    return aggregated_ranks

def save_pickle(data, file_path):
    """Save data to a pickle file."""
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def create_new_pickle_with_summed_sim_mat(original_file, summed_sim_mat, new_file_path):
    """Create a new pickle file with the summed sim_mat, keeping other elements the same."""
    # Load the original data
    data = load_pickle(original_file)
    
    # Replace the sim_mat with the summed_sim_mat
    data['sim_mat'] = summed_sim_mat
    
    # Save the updated data to a new pickle file
    save_pickle(data, new_file_path)

# Example usage
#pickle_files = ['../2test.pkl', '../test1.pkl', '../test2.pkl','../test4.pkl', '../test10.pkl', '../test11.pkl']  # Replace with the actual paths to your pickle files
pickle_files = ['../test11.pkl']
summed_sim_mat = sum_sim_mats(pickle_files)
#summed_sim_mat = rank_aggregation(pickle_files)
rel_matrix = pd.read_pickle('/home/wangxiaoqi/EK100_320p_15sec_30fps_libx264/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl')
vis_map, txt_map, avg_map = get_mAP(summed_sim_mat, rel_matrix)
print('mAP: V->T: {:.4f} T->V: {:.4f} AVG: {:.4f}'.format(vis_map, txt_map, avg_map))
vis_nDCG, txt_nDCG, avg_nDCG = get_nDCG(summed_sim_mat, rel_matrix)
print('nDCG: V->T: {:.4f} T->V: {:.4f} AVG: {:.4f}'.format(vis_nDCG, txt_nDCG, avg_nDCG))

# print("Summed sim_mat:")
# print(summed_sim_mat)

original_file = '../test.pkl'  # Use one of the original files as a template
new_file_path = '../new_test.pkl'  # Path to the new pickle file
#create_new_pickle_with_summed_sim_mat(original_file, summed_sim_mat, new_file_path)