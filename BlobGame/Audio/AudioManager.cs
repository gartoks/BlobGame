using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using Raylib_CsLo;
using System.Collections.Concurrent;

namespace BlobGame.Audio;
/// <summary>
/// Class for managing sounds and music.
/// </summary>
internal static class AudioManager {
    private const float MUSIC_VOLUME_MODIFIER = 0.4f;

    private static BlockingCollection<Action> SoundActionQueue { get; }

    private static BlockingCollection<Action> MusicActionQueue { get; }

    /// <summary>
    /// Dictionary of all currently playing sounds.
    /// </summary>
    private static ConcurrentDictionary<string, SoundResource> PlayingSounds { get; }
    /// <summary>
    /// Dictionary of all currently playing music.
    /// </summary>
    private static ConcurrentDictionary<string, MusicResource> PlayingMusic { get; }

    static AudioManager() {
        SoundActionQueue = new();
        MusicActionQueue = new();

        PlayingSounds = new();
        PlayingMusic = new();
    }

    /// <summary>
    /// Initializes the audio manager. Currently does nothing.
    /// </summary>
    internal static void Initialize() {
    }

    /// <summary>
    /// Loads audio manager resources. Currently does nothing.
    /// </summary>
    internal static void Load() {
    }

    /// <summary>
    /// Handles keeping track if currently playing sounds and music. Is called every frame.
    /// </summary>
    internal static void Update() {
        while (SoundActionQueue.TryTake(out Action action, 0)) {
            action.Invoke();
        }

        foreach (string item in PlayingSounds.Keys.ToList()) {

            Sound sound = PlayingSounds[item].Resource;
            if (Raylib.IsSoundPlaying(sound)) {
                // TODO: figure out if sound needs updating too somehow
            } else
                StopSound(item);
        }

        while (MusicActionQueue.TryTake(out Action action, 0)) {
            action.Invoke();
        }

        foreach (string item in PlayingMusic.Keys.ToList()) {
            Music music = PlayingMusic[item].Resource;
            if (Raylib.IsMusicStreamPlaying(music))
                Raylib.UpdateMusicStream(music);
            else
                StopMusic(item);
        }
    }

    /// <summary>
    /// Starts playing the sound with the given name. If the sound was playing already, it is restarted.
    /// </summary>
    /// <param name="name">The resource key of the sound to play.</param>
    internal static void PlaySound(string name) {
        if (PlayingSounds.ContainsKey(name))
            StopSound(name);

        SoundResource sound = ResourceManager.SoundLoader.Get(name);

        SoundActionQueue.Add(() => {
            PlayingSounds[name] = sound;
            Raylib.SetSoundVolume(sound.Resource, Application.Settings.SoundVolume / 100f);
            Raylib.PlaySound(sound.Resource);
        });
    }

    /// <summary>
    /// Stops the sound with the given name.
    /// </summary>
    /// <param name="name">The resource key of the sound to stop playing.</param>
    internal static void StopSound(string name) {
        if (!PlayingSounds.TryGetValue(name, out SoundResource? sound))
            return;

        SoundActionQueue.Add(() => {
            Raylib.StopSound(sound.Resource);
            PlayingSounds.Remove(name, out _);
        });
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
        if (Application.Settings.SoundVolume != v) {
            Application.Settings.SoundVolume = v;
            return;
        }

        SoundActionQueue.Add(() => {
            foreach (string item in PlayingSounds.Keys.ToList())
                Raylib.SetSoundVolume(PlayingSounds[item].Resource, v / 100f);
        });
    }

    /// <summary>
    /// Stops all currently playing sounds.
    /// </summary>
    internal static void ClearSounds() {
        while (SoundActionQueue.TryTake(out _)) ;

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

        MusicResource music = ResourceManager.MusicLoader.Get(name);

        MusicActionQueue.Add(() => {
            //Debug.WriteLine($"Playing {name} ({Raylib.GetMusicTimeLength(music.Resource)}s)");
            PlayingMusic[name] = music;
            Raylib.PlayMusicStream(music.Resource);
            Raylib.SetMusicVolume(music.Resource, MUSIC_VOLUME_MODIFIER * Application.Settings.MusicVolume / 100f);
        });
    }

    /// <summary>
    /// Stops the music with the given name.
    /// </summary>
    /// <param name="name">The resource key of the music to stop playing.</param>
    internal static void StopMusic(string name) {
        if (!PlayingMusic.TryGetValue(name, out MusicResource? music))
            return;

        MusicActionQueue.Add(() => {
            Raylib.StopMusicStream(music.Resource);
            PlayingMusic.Remove(name, out _);
        });
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
        if (Application.Settings.MusicVolume != v) {
            Application.Settings.MusicVolume = v;
            return;
        }

        MusicActionQueue.Add(() => {
            foreach (string item in PlayingMusic.Keys.ToList())
                Raylib.SetMusicVolume(PlayingMusic[item].Resource, MUSIC_VOLUME_MODIFIER * v / 100f);
        });
    }

    /// <summary>
    /// Stops all currently playing music.
    /// </summary>
    internal static void ClearMusic() {
        while (MusicActionQueue.TryTake(out _)) ;

        foreach (string item in PlayingMusic.Keys.ToList()) {
            StopMusic(item);
        }
    }
}
