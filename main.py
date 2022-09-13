import rain_filter as rf


def main():
    [Fs, data] = rf.load_file("./sounds/test.wav");
    rf.freq_analysis(data, Fs);
    data_filtered = rf.rainfilter(data, Fs, 9);
    rf.freq_analysis(data_filtered, Fs);
    rf.play_file(data, Fs, 2);
    rf.play_file(data_filtered, Fs, 2);
    print(data.shape[1])
    length = data.shape[0] / Fs
    print(data.shape[0])
    print(data.shape)
    print(length)

if __name__ == "__main__":
    main()