from scipy.io import loadmat

# 경로에 있는 .mat 파일 불러오기
mat_data = loadmat("/home/chan/jsdf_ws/install/data_sampling/share/data_sampling/data/points.mat")  # 경로를 실제 파일 경로로 바꾸세요
total_array = mat_data["total_array"]

# 배열 shape 출력
print("Shape of total_array:", total_array.shape)