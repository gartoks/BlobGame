using NAudio.Wave;

namespace BlobGame.Audio;
internal class AudioFile : IDisposable {
    protected static (float[] data, WaveFormat format) ReadWaveStream(WaveStream reader) {
        WaveFormat format = reader.WaveFormat;
        List<float> rawFile = new List<float>((int)reader.Length / 4);
        byte[] buffer = new byte[reader.WaveFormat.BlockAlign * reader.WaveFormat.SampleRate * reader.WaveFormat.Channels];

        int bytesRead = 0;
        while ((bytesRead = reader.Read(buffer, 0, buffer.Length)) > 0) {
            // Convert byte array to float array
            int floatsRead = bytesRead / 4;
            float[] floatBuffer = new float[floatsRead];
            for (int i = 0; i < floatsRead; i++) {
                floatBuffer[i] = BitConverter.ToSingle(buffer, i * 4);
            }
            rawFile.AddRange(floatBuffer.Take(floatsRead));
        }

        float[] data = rawFile.ToArray();

        return (data, format);
    }

    private WaveFormat Format { get; }
    private float[] Data { get; set; }

    protected AudioFile(float[] data, WaveFormat format) {
        Data = data;
        Format = format;
    }


    private bool disposedValue;

    // override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
    ~AudioFile() {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(disposing: false);
    }

    private void Dispose(bool disposing) {
        if (!disposedValue) {
            if (disposing) {
                // TODO: dispose managed state (managed objects)
            }

            // free unmanaged resources (unmanaged objects) and override finalizer
            // set large fields to null
            Data = null;

            disposedValue = true;
        }
    }

    public void Dispose() {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    internal sealed class AudioSampleProvider : ISampleProvider {
        private AudioFile Audio { get; }
        private long Position { get; set; }

        public WaveFormat WaveFormat => Audio.Format;

        internal AudioSampleProvider(AudioFile audio) {
            Audio = audio;
            Position = 0;
        }

        public int Read(float[] buffer, int offset, int count) {
            long remainingSamples = Audio.Data.Length - Position;
            long samplesToCopy = Math.Min(remainingSamples, count);
            Array.Copy(Audio.Data, Position, buffer, offset, samplesToCopy);
            Position += samplesToCopy;
            return (int)samplesToCopy;
        }
    }
}

