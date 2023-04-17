from matplotlib import pyplot as plt

def plot(data, fps_history, closed_eyes_frames):
    fig, axs = plt.subplot_mosaic([['a0', 'b'],['a1', 'b'],['a2', 'c'],['a3', 'd']], layout='constrained', figsize=(8, 8))
    fig.canvas.manager.set_window_title('Statistics')
    
    axs['a0'].plot([x[0] for x in data])
    axs['a0'].set_title('Pose')

    axs['a1'].plot([x[1] for x in data])
    axs['a1'].set_title('CEF_COUNTER')

    axs['a2'].plot([x[2] for x in data])
    axs['a2'].set_title('TOTAL_BLINKS')

    axs['a3'].plot([x[3] for x in data])
    axs['a3'].set_title('Eye Position')

    # pose array is something like ["neutro", "feliz", "tedio"]
    # converting pose data so we can plot in pie chart
    poses = [x[0] for x in data]
    labels = []

    for pose in poses:
        if pose not in labels:
            labels.append(pose)

    poses_to_plot = [poses.count(x) for x in labels]

    axs['b'].pie(poses_to_plot, labels=labels, autopct='%1.1f%%')
    axs['b'].set_title('Pose ratio')

    # converting eye position data so we can plot in pie chart
    eye_positions = [x[3] for x in data]
    labels = []

    for eye_position in eye_positions:
        if eye_position not in labels:
            labels.append(eye_position)

    eye_positions_to_plot = [eye_positions.count(x) for x in labels]

    axs['c'].pie(eye_positions_to_plot, labels=labels, autopct='%1.1f%%')
    axs['c'].set_title('Eye Position ratio')

    mean_fps = sum(fps_history) / len(fps_history)

    # show closed eyes frames value in a table
    CE_time = round(closed_eyes_frames / mean_fps, 2)
    axs['d'].axis('off')
    table = axs['d'].table(cellText=[['closed_eyes_frames'], [f"{closed_eyes_frames} frames"], ['average estimated CE time'], [f"{CE_time} seconds"]], cellLoc='center', loc='center')
    table.auto_set_column_width(True)
    table.set_fontsize(18)

    # making even lines bold
    for (row, col), cell in table.get_celld().items():
        if (row == 0) or (row % 2 == 0):
            cell.set_text_props(weight='bold')

    for cell in table._cells.values():
         # height
        cell.set_height(0.25)

    plt.show()