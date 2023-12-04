using NAudio.Wave;

namespace BlobGame.Audio;
internal sealed class Sound : AudioFile {
    public static Sound Create(Stream stream) {
        using WaveFileReader reader = new WaveFileReader(stream);
        (float[] data, WaveFormat format) fileData = ReadWaveStream(reader);
        return new Sound(fileData.data, fileData.format);
    }

    private Sound(float[] data, WaveFormat format)
        : base(data, format) {
    }
}
