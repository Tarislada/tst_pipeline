import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the data from the npy file
# file_path = r"C:\1_Kim_Lab_2023\AMOMA\TEST\enclosed_pixels.npy"  # Replace with your file path
# mouse_data_raw = np.load(file_path)


######################################################################################################

### STEP 1 Plotting different mathematical thingies

# # Load enclosed pixels array
arr = np.load('/home/tarislada/Documents/Extra_python_projects/Natalie/output_data.npy')
if arr.ndim == 2 and arr.shape[1] >= 1:
    enclosed_pixels = arr[:, 0]
else:
    enclosed_pixels = arr
# fps = 30
# n_seconds = len(enclosed_pixels) // fps

# # Lists to store computed features
# mean_values = []
# std_values = []
# max_values = []
# min_values = []
# sum_abs_diffs = []

# for sec in range(n_seconds):
#     start = sec * fps
#     end = start + fps
#     bin_data = enclosed_pixels[start:end]
    
#     # Compute statistical features
#     mean_val = np.mean(bin_data)
#     std_val = np.std(bin_data)
#     max_val = np.max(bin_data)
#     min_val = np.min(bin_data)
#     abs_diffs = np.abs(np.diff(bin_data))
#     sum_abs_diff = np.sum(abs_diffs)
    
#     mean_values.append(mean_val)
#     std_values.append(std_val)
#     max_values.append(max_val)
#     min_values.append(min_val)
#     sum_abs_diffs.append(sum_abs_diff)

# # Convert lists to numpy arrays for further analysis if needed
# mean_values = np.array(mean_values)
# std_values = np.array(std_values)
# max_values = np.array(max_values)
# min_values = np.array(min_values)
# sum_abs_diffs = np.array(sum_abs_diffs)

# # Plotting these features for visual inspection
# time_axis = np.arange(n_seconds)

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 2, 1)
# plt.plot(time_axis, mean_values, label='Mean')
# plt.title('Mean Enclosed Pixels (per second)')
# plt.xlabel('Time (s)')
# plt.legend()

# plt.subplot(2, 2, 2)
# plt.plot(time_axis, std_values, color='orange', label='Standard Deviation')
# plt.title('Std Dev of Enclosed Pixels (per second)')
# plt.xlabel('Time (s)')
# plt.legend()

# plt.subplot(2, 2, 3)
# plt.plot(time_axis, sum_abs_diffs, color='green', label='Sum of Abs Differences')
# plt.title('Sum of Absolute Differences (per second)')
# plt.xlabel('Time (s)')
# plt.legend()

# plt.subplot(2, 2, 4)
# plt.plot(time_axis, max_values - min_values, color='red', label='Range (Max - Min)')
# plt.title('Range of Enclosed Pixels (per second)')
# plt.xlabel('Time (s)')
# plt.legend()

# plt.tight_layout()
# # plt.show()



######################################################################################################

### STEP 2 Mobile vs Immobile analysis

### STEP 2.1 Min Max STD


# 250819 The code was added so that the std would be calculated 
# based on the min max of each processed mouse video. This was done so that the threshold
# could be the same across all mouse videos. Previously the threshold had to be adjusted per 
# mouse video since it was based on the overal std. Some mice's enclosed pixel range was larger than others.

# # Load the enclosed pixels array
# enclosed_pixels = np.load(r"C:\1_Kim_Lab_2023\AMOMA\KHC2_Segments\KHC2_M4_post\enclosed_pixels.npy")

# fps = 30
# n_seconds = len(enclosed_pixels) // fps

# # Calculate standard deviation for each 1-second bin
# std_values = []
# time_values = []
# for sec in range(n_seconds):
#     start = sec * fps
#     end = start + fps
#     bin_data = enclosed_pixels[start:end]
#     std_values.append(np.std(bin_data))
#     time_values.append(sec)  # Time in seconds

# std_values = np.array(std_values)
# time_values = np.array(time_values)

# # 🔹 Normalize std values per video
# eps = 1e-9  # small constant to avoid division by zero
# std_norm = (std_values - np.min(std_values)) / (np.max(std_values) - np.min(std_values) + eps)

# # 🔹 Apply global threshold (same for all videos)
# threshold = 0.1
# state = np.where(std_norm < threshold, 0, 1)

# # Create a DataFrame for convenience and filter to 180 seconds
# df = pd.DataFrame({
#     "Time (s)": time_values,
#     "Standard Deviation": std_values,
#     "Normalized STD": std_norm,
#     "State": state
# })
# df = df[df["Time (s)"] < 180]

# # Refine state so immobility must last >= 1 second (i.e., at least 2 bins)
# refined_state = state.copy()
# for i in range(len(state) - 1):
#     if state[i] == 0 and state[i + 1] == 1:
#         refined_state[i] = 1  # remove single-bin immobility

# # Function to compute contiguous intervals (same as before)
# def get_contiguous_intervals(time_array, state_array):
#     intervals = []
#     current_state = state_array[0]
#     start_time = time_array[0]
#     for t, s in zip(time_array[1:], state_array[1:]):
#         if s != current_state:
#             intervals.append((start_time, t, current_state))
#             start_time = t
#             current_state = s
#     intervals.append((start_time, time_array[-1] + 1, current_state))
#     return intervals

# intervals = get_contiguous_intervals(time_values, refined_state)


# # Setup the plot
# fig, ax = plt.subplots(figsize=(18, 4))

# # Plot background colored bars for each contiguous state interval
# for start, end, s in intervals:
#     color = "#E09E4E" if s == 0 else "#315F91"  # 0: orange (immobile), 1: blue (mobile)
#     ax.axvspan(start, end, facecolor=color, zorder=0)

# # 🔹 Plot the normalized std line instead of raw std
# ax.plot(df["Time (s)"], df["Normalized STD"], 'k-', linewidth=2, zorder=5, label="Normalized STD")

# # Set x-axis limits and ticks (max 180 seconds, with 10 second bins)
# ax.set_xlim(0, 180)
# ax.set_xticks(np.arange(0, 181, 10))
# plt.yticks([])
# plt.xticks(np.arange(0, 181, 10), fontsize=20, fontweight='bold')

# # Customize plot appearance
# ax.set_title("Automated Mobility Analysis (Normalized STD)", fontsize=24, fontweight='bold', pad=40)
# ax.set_xlabel("Time (s)", fontsize=20, fontweight='bold', labelpad=40)
# ax.set_xlim(0, df["Time (s)"].max() + 1)
# ax.tick_params(axis='x', labelsize=20, width=4, length=10)
# ax.tick_params(axis='y', labelsize=20, width=4, length=10)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_linewidth(4)
# ax.set_ylabel("Normalized STD", fontsize=20, fontweight='bold', labelpad=40)

# plt.tight_layout()
# # plt.show()


######################################################################################################

### STEP 2.2 Variance
# Load the array: supports both (T,) legacy and (T,2) [area, similarity]
arr = np.load('/home/tarislada/Documents/Extra_python_projects/Natalie/output_data.npy')
if arr.ndim == 1:
    area = arr.astype(np.float32)
    sim  = np.zeros_like(area, dtype=np.float32)    # no similarity -> behave like before
else:
    area = arr[:, 0].astype(np.float32)
    sim  = arr[:, 1].astype(np.float32) if arr.shape[1] > 1 else np.zeros_like(area, dtype=np.float32)
    sim  = np.clip(sim, 0.0, 1.0)

# Similarity -> weight (only used for classification, NOT for plotting)
# beta, w_min = 0.31, 0.75
beta, w_min = 0.01, 0.1
w = np.clip(1.0 - beta * sim, w_min, 1.0)

fps = 30
n_seconds = len(area) // fps

# --- per-second variance from raw area (for plotting)
var_raw, time_values = [], []
for sec in range(n_seconds):
    s, e = sec * fps, sec * fps + fps
    var_raw.append(np.var(area[s:e]))
    time_values.append(sec)
var_raw = np.array(var_raw, dtype=np.float32)
time_values = np.array(time_values, dtype=np.int32)

# --- per-second variance from weighted area (for classification only)
var_eff = []
for sec in range(n_seconds):
    s, e = sec * fps, sec * fps + fps
    var_eff.append(np.var((area[s:e]) * (w[s:e])))  # This is where beta and w_min matter
var_eff = np.array(var_eff, dtype=np.float32)

# Normalize BOTH using raw min/max so the visual scale and threshold semantics stay fixed
eps = 1e-9
raw_min, raw_max = np.min(var_raw), np.max(var_raw)
den = (raw_max - raw_min + eps)
var_norm_plot = (var_raw - raw_min) / den        # plotted line (unchanged)
var_norm_for_state = (var_eff - raw_min) / den   # only for state thresholding

# Global threshold and state (coloring) use the similarity-adjusted series
threshold = 0.02
state = np.where(var_norm_for_state < threshold, 0, 1)  # 0: immobile (orange), 1: mobile (blue)

# Keep first 180 s like before
df = pd.DataFrame({
    "Time (s)": time_values,
    "Variance": var_raw,
    "Normalized VAR": var_norm_plot,
    "State": state
})
df = df[df["Time (s)"] < 180]

# Refine: immobility must last >= 1 s (at least 2 bins)
refined_state = state.copy()
for i in range(len(state) - 1):
    if state[i] == 0 and state[i + 1] == 1:
        refined_state[i] = 1

def get_contiguous_intervals(time_array, state_array):
    intervals = []
    current_state = state_array[0]
    start_time = time_array[0]
    for t, s in zip(time_array[1:], state_array[1:]):
        if s != current_state:
            intervals.append((start_time, t, current_state))
            start_time = t
            current_state = s
    intervals.append((start_time, time_array[-1] + 1, current_state))
    return intervals

intervals = get_contiguous_intervals(time_values, refined_state)

# ---- Plot: line is raw (unchanged), background coloring from similarity-adjusted state
fig, ax = plt.subplots(figsize=(18, 4))
for start, end, s in intervals:
    color = "#E09E4E" if s == 0 else "#315F91"
    ax.axvspan(start, end, facecolor=color, zorder=0)

ax.plot(df["Time (s)"], df["Normalized VAR"], 'k-', linewidth=2, zorder=5, label="Normalized VAR")

ax.set_xlim(0, 180)
ax.set_xticks(np.arange(0, 181, 10))
plt.yticks([])
plt.xticks(np.arange(0, 181, 10), fontsize=20, fontweight='bold')

ax.set_title("Automated Mobility Analysis (Normalized Variance)", fontsize=24, fontweight='bold', pad=40)
ax.set_xlabel("Time (s)", fontsize=20, fontweight='bold', labelpad=40)
ax.set_xlim(0, df["Time (s)"].max() + 1)
ax.tick_params(axis='x', labelsize=20, width=4, length=10)
ax.tick_params(axis='y', labelsize=20, width=4, length=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_linewidth(4)
ax.set_ylabel("Normalized Variance", fontsize=20, fontweight='bold', labelpad=40)

plt.tight_layout()
plt.show()


######################################################################################################

### STEP 3 OLD: STD based mobility (not normalized)

# # Load the enclosed pixels array
# enclosed_pixels = np.load(r"C:\1_Kim_Lab_2023\AMOMA\250819_KHC2_Segments\250825_KHC2_pre_M3\enclosed_pixels.npy")

# fps = 30
# n_seconds = len(enclosed_pixels) // fps

# # Calculate standard deviation for each 1-second bin
# std_values = []
# time_values = []
# for sec in range(n_seconds):
#     start = sec * fps
#     end = start + fps
#     bin_data = enclosed_pixels[start:end]
#     std_values.append(np.std(bin_data))
#     time_values.append(sec)  # Time in seconds

# std_values = np.array(std_values)
# time_values = np.array(time_values)

# # Determine mobility state: 0 = Immobile (std < 200), 1 = Mobile (std >= 200)
# threshold = 750
# state = np.where(std_values < threshold, 0, 1)

# # Create a DataFrame for convenience and filter to 180 seconds
# df = pd.DataFrame({
#     "Time (s)": time_values,
#     "Standard Deviation": std_values,
#     "State": state
# })
# df = df[df["Time (s)"] < 180]

# # Function to compute contiguous intervals for the binary state
# def get_contiguous_intervals(time_array, state_array):
#     intervals = []
#     current_state = state_array[0]
#     start_time = time_array[0]
#     for t, s in zip(time_array[1:], state_array[1:]):
#         if s != current_state:
#             # end time is current t (each bin is 1 sec, so interval [start, t))
#             intervals.append((start_time, t, current_state))
#             start_time = t
#             current_state = s
#     intervals.append((start_time, time_array[-1] + 1, current_state))
#     return intervals

# intervals = get_contiguous_intervals(time_values, state)

# # Setup the plot
# fig, ax = plt.subplots(figsize=(18, 4))

# # Plot background colored bars for each contiguous state interval
# for start, end, s in intervals:
#     color = "#E09E4E" if s == 0 else "#315F91"  # 0: orange (immobile), 1: blue (mobile)
#     ax.axvspan(start, end, facecolor=color, zorder=0)

# # Plot the standard deviation line on top (with a higher zorder)
# ax.plot(df["Time (s)"], df["Standard Deviation"], 'k-', linewidth=2, zorder=5, label="Standard Deviation")

# # Set x-axis limits and ticks (max 180 seconds, with 10 second bins)
# ax.set_xlim(0, 180)
# ax.set_xticks(np.arange(0, 181, 10))
# plt.yticks([])
# plt.xticks(np.arange(0, 181, 10), fontsize=20, fontweight='bold')

# # Customize plot appearance
# ax.set_title("Automated Mobility Analysis Based on Standard Deviation", fontsize=24, fontweight='bold', pad=40)
# ax.set_xlabel("Time (s)", fontsize=20, fontweight='bold', labelpad=40)
# ax.set_xlim(0, df["Time (s)"].max() + 1)
# ax.tick_params(axis='x', labelsize=20, width=4, length=10)
# ax.tick_params(axis='y', labelsize=20, width=4, length=10)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_linewidth(4)
# ax.set_ylabel("Standard Deviation", fontsize=20, fontweight='bold', labelpad=40)

# # Optional: Add a legend using custom patches
# import matplotlib.patches as mpatches
# immobile_patch = mpatches.Patch(color="#E09E4E", alpha=0.3, label="Immobile")
# mobile_patch = mpatches.Patch(color="#315F91", alpha=0.3, label="Mobile")
# ax.legend(handles=[immobile_patch, mobile_patch], loc="upper left", bbox_to_anchor=(0, 1), ncol=2)
 
# plt.tight_layout()
# plt.show()
# ######################################################################################################




######################################################################################################


# # Define thresholds (these values are examples; you'll need to tune them)
# std_threshold = 100        # example threshold for standard deviation
# abs_diff_threshold = 5000   # example threshold for sum of abs differences

# mobility_state = []  # True for immobile, False for mobile

# for std_val, abs_diff in zip(std_values, sum_abs_diffs):
#     # A simple rule: if both std dev and abs difference are low, it's immobile.
#     if std_val < std_threshold and abs_diff < abs_diff_threshold:
#         mobility_state.append(True)  # Immobile
#     else:
#         mobility_state.append(False)  # Mobile

# # Plot the mobility state alongside one of your features for comparison
# plt.figure(figsize=(10, 4))
# plt.plot(time_axis, std_values, label='Std Dev')
# plt.plot(time_axis, mobility_state, 'r.-', label='Immobile (True=1, Mobile=0)')
# plt.xlabel('Time (s)')
# plt.title('Automated Mobility State Based on Std Dev')
# plt.legend()
# plt.show()


