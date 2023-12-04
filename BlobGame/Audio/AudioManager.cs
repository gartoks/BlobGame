using BlobGame.App;
using BlobGame.ResourceHandling;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using SimpleGL.Util;
using System.Collections.Concurrent;
using static BlobGame.Audio.AudioFile;

namespace BlobGame.Audio;
/// <summary>
/// Class for managing sounds and music.
/// </summary>
internal static class AudioManager {
    private const int SAMPLE_RATE = 44100;
    private const int CHANNELS = 2;
    //private const float MUSIC_VOLUME_MODIFIER = 0.4f;

    //private static BlockingCollection<Action> SoundActionQueue { get; }
    //private static BlockingCollection<Action> MusicActionQueue { get; }

    /// <summary>
    /// Dictionary of all currently playing sounds.
    /// </summary>
    private static ConcurrentDictionary<string, ISampleProvider> PlayingSounds { get; }
    /// <summary>
    /// Dictionary of all currently playing music.
    /// </summary>
    private static ConcurrentDictionary<string, ISampleProvider> PlayingMusic { get; }

    private static IWavePlayer SoundPlayer { get; }
    private static MixingSampleProvider SoundMixer { get; }

    private static IWavePlayer MusicPlayer { get; }
    private static MixingSampleProvider MusicMixer { get; }

    static AudioManager() {
        //SoundActionQueue = new();
        //MusicActionQueue = new();

        PlayingSounds = new();
        PlayingMusic = new();

        SoundPlayer = new WaveOutEvent();
        SoundMixer = new MixingSampleProvider(WaveFormat.CreateIeeeFloatWaveFormat(SAMPLE_RATE, CHANNELS));

        MusicPlayer = new WaveOutEvent();
        MusicMixer = new MixingSampleProvider(WaveFormat.CreateIeeeFloatWaveFormat(SAMPLE_RATE, CHANNELS));
    }

    /// <summary>
    /// Initializes the audio manager. Currently does nothing.
    /// </summary>
    internal static void Initialize() {
        SoundPlayer.Init(SoundMixer);
        MusicPlayer.Init(MusicMixer);

        SoundPlayer.PlaybackStopped += OnSoundPlaybackSStopped;
        SoundMixer.MixerInputEnded += OnSoundMixerInputEnded;
    }

    private static void OnSoundMixerInputEnded(object? sender, SampleProviderEventArgs e) {
    }

    private static void OnSoundPlaybackSStopped(object? sender, StoppedEventArgs e) {
    }

    /// <summary>
    /// Loads audio manager resources. Currently does nothing.
    /// </summary>
    internal static void Load() {
        SoundPlayer.Play();
        MusicPlayer.Play();
    }

    internal static void Unload() {
        SoundPlayer.PlaybackStopped -= OnSoundPlaybackSStopped;
        SoundMixer.MixerInputEnded -= OnSoundMixerInputEnded;

        SoundPlayer.Stop();
        MusicPlayer.Stop();

        PlayingSounds.Clear();
        PlayingMusic.Clear();

        SoundPlayer.Dispose();
        MusicPlayer.Dispose();
    }

    /// <summary>
    /// Handles keeping track if currently playing sounds and music. Is called every frame.
    /// </summary>
    internal static void Update() {
        //while (SoundActionQueue.TryTake(out Action? action, 0)) {
        //    action!.Invoke();
        //}

        /*foreach (string item in PlayingSounds.Keys.ToList()) {
            Sound sound = PlayingSounds[item].Resource;
            if (Raylib.IsSoundPlaying(sound)) {
                // TODO: figure out if sound needs updating too somehow
            } else
                StopSound(item);
        }*/

        //while (MusicActionQueue.TryTake(out Action action, 0)) {
        //    action.Invoke();
        //}

        /*foreach (string item in PlayingMusic.Keys.ToList()) {
            Music music = PlayingMusic[item].Resource;
            if (Raylib.IsMusicStreamPlaying(music))
                Raylib.UpdateMusicStream(music);
            else
                StopMusic(item);
        }*/
    }

    /// <summary>
    /// Starts playing the sound with the given name. If the sound was playing already, it is restarted.
    /// </summary>
    /// <param name="name">The resource key of the sound to play.</param>
    internal static void PlaySound(string name) {
        if (PlayingSounds.ContainsKey(name))
            StopSound(name);

        if (!ResourceManager.SoundLoader.TryGetResource(name, out Sound? sound)) {
            Log.WriteLine($"Sound {name} not found.", eLogType.Warning);
            return;
        }

        //SoundActionQueue.Add(() => {  // TODO test if works without
        PlayingSounds[name] = ConvertChannelCount(SoundMixer, new AudioSampleProvider(sound!));
        SoundMixer.AddMixerInput(PlayingSounds[name]);
        //});
    }

    /// <summary>
    /// Stops the sound with the given name.
    /// </summary>
    /// <param name="name">The resource key of the sound to stop playing.</param>
    internal static void StopSound(string name) {
        if (!PlayingSounds.TryGetValue(name, out ISampleProvider? sound))
            return;

        //SoundActionQueue.Add(() => {  // TODO test if works without
        SoundMixer.RemoveMixerInput(sound!);
        PlayingSounds.Remove(name, out _);
        //});
    }

    /// <summary>
    /// Checks if the sound with the given name is currently playing.
    /// </summary>
    /// <param name="name">The resource key of the sound.</param>
    /// <returns>Return true if the sound is playing; false otherwise.</returns>
    internal static bool IsSoundPlaying(string name) {
        return PlayingSounds.ContainsKey(name);
    }

    /// <summary>
    /// Updates all playing sounds to the new volume.
    /// This also updates the sound volume setting if necessary.
    /// </summary>
    /// <param name="v">Volume value. Must be in [0, 100]</param>
    internal static void SetSoundVolume(int v) {
        if (GameApplication.Settings.SoundVolume != v) {
            GameApplication.Settings.SoundVolume = v;
            return;
        }

        //SoundActionQueue.Add(() => {  // TODO test if works without
        SoundPlayer.Volume = v / 100f;
        //});
    }

    /// <summary>
    /// Stops all currently playing sounds.
    /// </summary>
    internal static void ClearSounds() {
        //while (SoundActionQueue.TryTake(out _)) ;

        foreach (string item in PlayingSounds.Keys.ToList()) {
            StopSound(item);
        }
    }

    /// <summary>
    /// Starts playing the music with the given name. If the music was playing already, it is restarted.
    /// </summary>
    /// <param name="name">The resource key of the music to play.</param>
    internal static void PlayMusic(string name) {
        if (PlayingMusic.ContainsKey(name))
            StopMusic(name);

        if (!ResourceManager.MusicLoader.TryGetResource(name, out Music? music)) {
            Log.WriteLine($"Music {name} not found.", eLogType.Warning);
            return;
        }

        //MusicActionQueue.Add(() => {  // TODO test if works without
        //Debug.WriteLine($"Playing {name} ({Raylib.GetMusicTimeLength(music.Resource)}s)");
        PlayingMusic[name] = ConvertChannelCount(MusicMixer, new AudioSampleProvider(music!));
        MusicMixer.RemoveAllMixerInputs();
        MusicMixer.AddMixerInput(PlayingMusic[name]);
        //});
    }

    /// <summary>
    /// Stops the music with the given name.
    /// </summary>
    /// <param name="name">The resource key of the music to stop playing.</param>
    internal static void StopMusic(string name) {
        if (!PlayingMusic.TryGetValue(name, out ISampleProvider? music))
            return;

        //MusicActionQueue.Add(() => {  // TODO test if works without
        MusicMixer.RemoveMixerInput(music!);
        PlayingMusic.Remove(name, out _);
        //});
    }

    /// <summary>
    /// Checks if the music with the given name is currently playing.
    /// </summary>
    /// <param name="name">The resource key of the music.</param>
    /// <returns>Return true if the music is playing; false otherwise.</returns>
    internal static bool IsMusicPlaying(string name) {
        return PlayingMusic.ContainsKey(name);
    }

    /// <summary>
    /// Updates all playing music to the new volume.
    /// This also updates the music volume setting if necessary.
    /// </summary>
    /// <param name="v">Volume value. Must be in [0, 100]</param>
    internal static void SetMusicVolume(int v) {
        if (GameApplication.Settings.MusicVolume != v) {
            GameApplication.Settings.MusicVolume = v;
            return;
        }

        //MusicActionQueue.Add(() => {  // TODO test if works without
        MusicPlayer.Volume = v / 100f;
        //});
    }

    /// <summary>
    /// Stops all currently playing music.
    /// </summary>
    internal static void ClearMusic() {
        //while (MusicActionQueue.TryTake(out _)) ;

        foreach (string item in PlayingMusic.Keys.ToList()) {
            StopMusic(item);
        }
    }

    private static ISampleProvider ConvertChannelCount(MixingSampleProvider mixer, ISampleProvider input) {
        if (input.WaveFormat.Channels == mixer.WaveFormat.Channels)
            return input;

        if (input.WaveFormat.Channels == 1 && mixer.WaveFormat.Channels == 2)
            return new MonoToStereoSampleProvider(input);

        throw new ArgumentException("Audio channel count mismatch");
    }
}
