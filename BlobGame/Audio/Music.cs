using NAudio.Wave;

namespace BlobGame.Audio;
internal class Music : AudioFile {
    public static Music Create(Stream stream) {
        using Mp3FileReader reader = new Mp3FileReader(stream);
        (float[] data, WaveFormat format) fileData = ReadWaveStream(reader);
        return new Music(fileData.data, fileData.format);
    }

    private Music(float[] data, WaveFormat format)
        : base(data, format) {
    }
}
