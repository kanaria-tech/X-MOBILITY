import pandas as pd


file_path = ("/home/ozgur/Desktop/Others/Data/x_mobility_isaac_sim_nav2_100k/"
             "afm_isaac_sim_nav2_100k/data/train/sceneblox_warehouse_style_0_id_0/goal_0000_0hz.pqt")
df = pd.read_parquet(file_path)
print(df)
