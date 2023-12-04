using OpenTK.Mathematics;
using OpenTK.Windowing.GraphicsLibraryFramework;
using SimpleGL;
using SimpleGL.Graphics.GLHandling;
using static BlobGame.App.Input;

namespace BlobGame.App;
/// <summary>
/// Static class to control the game's input and convert them to hotkey and mouse states.
/// </summary>
public static class Input {
    /// <summary>
    /// The interaction state any interaction can be in.
    /// </summary>
    public enum eInteractionState {
        Up, Pressed, Down, Released
    };

    /// <summary>
    /// Keeps track of the state of all mouse buttons.
    /// </summary>
    private static Dictionary<MouseButton, eInteractionState> MouseButtonStates { get; }
    /// <summary>
    /// Keeps track of all hotkeys.
    /// </summary>
    private static Dictionary<string, Hotkey> Hotkeys { get; }

    /// <summary>
    /// Whether the mouse was handled by the game this frame. If true, the mouse was handled and no other game should handle it.
    /// </summary>
    public static Dictionary<MouseButton, bool> WasMouseHandled { get; }

    /// <summary>
    /// Static constructor to initialize the mouse button and hotkey states.
    /// </summary>
    static Input() {
        MouseButtonStates = new Dictionary<MouseButton, eInteractionState>();
        MouseButtonStates[MouseButton.Left] = eInteractionState.Up;
        MouseButtonStates[MouseButton.Right] = eInteractionState.Up;
        MouseButtonStates[MouseButton.Middle] = eInteractionState.Up;

        WasMouseHandled = new Dictionary<MouseButton, bool>();
        foreach (MouseButton mb in Enum.GetValues<MouseButton>())
            WasMouseHandled[mb] = false;

        Hotkeys = new Dictionary<string, Hotkey>();
    }

    /// <summary>
    /// Initializes the input. Currently does nothing.
    /// </summary>
    internal static void Initialize() {
    }

    /// <summary>
    /// Method to load any files or resources. Currently not needed.
    /// </summary>
    internal static void Load() {
    }

    /// <summary>
    /// Called every frame to read and update the input states, as well as apply states to mouse buttons and hotkeys.
    /// </summary>
    internal static void Update() {
        MouseState mouseState = Application.Window.MouseState;
        Vector2 mPosDelta = mouseState.Delta;
        foreach (MouseButton mbs in MouseButtonStates.Keys) {
            if (mouseState.IsButtonReleased(mbs))
                MouseButtonStates[mbs] = eInteractionState.Released;
            else if (mouseState.IsButtonPressed(mbs))
                MouseButtonStates[mbs] = eInteractionState.Pressed;
            else if (mouseState.IsButtonDown(mbs))
                MouseButtonStates[mbs] = eInteractionState.Down;
            else
                MouseButtonStates[mbs] = eInteractionState.Up;
        }

        KeyboardState keyboardState = Application.Window.KeyboardState;
        foreach (Hotkey hotkey in Hotkeys.Values) {
            Keys ks = hotkey.PrimaryKey;
            eInteractionState state = hotkey.State is eInteractionState.Released or eInteractionState.Up ? eInteractionState.Up : eInteractionState.Released;

            bool allModifiersDown = hotkey.Modifiers.All(keyboardState.IsKeyDown);
            if (allModifiersDown) {
                if (keyboardState.IsKeyReleased(ks))
                    state = eInteractionState.Released;
                else if (keyboardState.IsKeyPressed(ks))
                    state = eInteractionState.Pressed;
                else if (keyboardState.IsKeyDown(ks))
                    state = eInteractionState.Down;
                else
                    state = eInteractionState.Up;
            }
            hotkey.State = state;
        }
    }

    /// <summary>
    /// Registeres a new hotkey with the given name, primary key, and modifiers.
    /// </summary>
    /// <param name="key">The unique hotkey identifier</param>
    /// <param name="primaryKey">The primary key of the hotkey. This controls the state.</param>
    /// <param name="modifiers">List of key modifiers that are required to be held down in addition to the primary key. Such as Ctrl, Shift, Alt.</param>
    public static void RegisterHotkey(string key, Keys primaryKey, params Keys[] modifiers) {
        Hotkeys.Add(key, new Hotkey(key, primaryKey, modifiers));
    }

    /// <summary>
    /// Unregisteres the hotkey with the given name.
    /// </summary>
    /// <param name="key"></param>
    public static void UnregisterHotkey(string key) {
        Hotkeys.Remove(key);
    }

    /// <summary>
    /// Gets the current position of the mouse in viewport coordinates.
    /// </summary>
    /// <returns></returns>
    public static Vector2 GetMousePosition() {
        Vector2 rawMPos = Application.Window.MouseState.Position;

        return rawMPos / Application.Window.Size * GLHandler.Viewport.Size; // TODO TEST
    }

    /// <summary>
    /// Gets the state of the given mouse button.
    /// </summary>
    /// <param name="button"></param>
    /// <returns></returns>
    public static eInteractionState GetMouseButtonState(MouseButton button) {
        return MouseButtonStates[button];
    }

    /// <summary>
    /// Returns whether the given hotkey is active. This means the primary key is releaed and all modifiers are pressed down.
    /// </summary>
    /// <param name="hotkey"></param>
    /// <returns></returns>
    public static bool IsHotkeyActive(string hotkey) {
        return Hotkeys[hotkey].State is eInteractionState.Released;
    }

    /// <summary>
    /// Returns whether the given hotkey is pressed. This means the primary key is pressed down and all modifiers are pressed down.
    /// </summary>
    /// <param name="hotkey"></param>
    /// <returns></returns>
    public static bool IsHotkeyDown(string hotkey) {
        return Hotkeys[hotkey].State is eInteractionState.Pressed or eInteractionState.Down;
    }

    /// <summary>
    /// Returns whether the given hotkey is active. This means the primary key is released and all modifiers are pressed down.
    /// </summary>
    /// <param name="button"></param>
    /// <returns></returns>
    public static bool IsMouseButtonActive(MouseButton button) {
        return !WasMouseHandled[button] && MouseButtonStates[button] == eInteractionState.Released;
    }

    /// <summary>
    /// Returns whether the given mouse button is pressed.
    /// </summary>
    /// <param name="button"></param>
    /// <returns></returns>
    public static bool IsMouseButtonDown(MouseButton button) {
        return MouseButtonStates[button] is eInteractionState.Pressed or eInteractionState.Down;
    }
}

/// <summary>
/// Helper class to keep track of hotkey data.
/// </summary>
internal class Hotkey {
    public string Key { get; }
    public Keys PrimaryKey { get; }
    public IReadOnlyList<Keys> Modifiers { get; }
    public eInteractionState State { get; set; }

    public Hotkey(string name, Keys primaryKey, params Keys[] modifiers) {
        Key = name;
        PrimaryKey = primaryKey;
        Modifiers = modifiers;
        State = eInteractionState.Up;
    }

    public override string ToString() => $"{Key} {PrimaryKey} {State}";
}
