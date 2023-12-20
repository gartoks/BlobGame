using Raylib_CsLo;
using System.Numerics;
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
        MouseButtonStates[MouseButton.MOUSE_BUTTON_LEFT] = eInteractionState.Up;
        MouseButtonStates[MouseButton.MOUSE_BUTTON_RIGHT] = eInteractionState.Up;
        MouseButtonStates[MouseButton.MOUSE_BUTTON_MIDDLE] = eInteractionState.Up;

        WasMouseHandled = new Dictionary<MouseButton, bool>();

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
        Vector2 mPosDelta = Raylib.GetMouseDelta();
        foreach (MouseButton mbs in MouseButtonStates.Keys) {
            if (Raylib.IsMouseButtonDown(mbs)) {
                if (MouseButtonStates[mbs] == eInteractionState.Up)
                    MouseButtonStates[mbs] = eInteractionState.Pressed;
                else
                    MouseButtonStates[mbs] = eInteractionState.Down;
            } else if (Raylib.IsMouseButtonUp(mbs)) {
                if (MouseButtonStates[mbs] == eInteractionState.Down)
                    MouseButtonStates[mbs] = eInteractionState.Released;
                else {
                    WasMouseHandled[mbs] = false;
                    MouseButtonStates[mbs] = eInteractionState.Up;
                }
            }
        }

        foreach (Hotkey hotkey in Hotkeys.Values) {
            eInteractionState state = hotkey.State is eInteractionState.Released or eInteractionState.Up ? eInteractionState.Up : eInteractionState.Released;

            bool allModifiersDown = hotkey.Modifiers.All(Raylib.IsKeyDown);
            if (allModifiersDown) {
                if (hotkey.PrimaryKey is KeyboardKey key) {
                    if (Raylib.IsKeyPressed(key))
                        state = eInteractionState.Pressed;
                    else if (Raylib.IsKeyDown(key))
                        state = eInteractionState.Down;
                    else if (Raylib.IsKeyReleased(key))
                        state = eInteractionState.Released;
                } else if (hotkey.PrimaryKey is MouseButton button) {
                    if (Raylib.IsMouseButtonPressed(button))
                        state = eInteractionState.Pressed;
                    else if (Raylib.IsMouseButtonDown(button))
                        state = eInteractionState.Down;
                    else if (Raylib.IsMouseButtonReleased(button))
                        state = eInteractionState.Released;
                }

            }
            hotkey.State = state;
        }
    }

    public static Vector2 ScreenToWorld(Vector2 screenPos) {
        return new Vector2(
            screenPos.X / Application.WorldToScreenMultiplierX,
            screenPos.Y / Application.WorldToScreenMultiplierY);
    }

    /// <summary>
    /// Registeres a new hotkey with the given name, primary key, and modifiers.
    /// </summary>
    /// <param name="key">The unique hotkey identifier</param>
    /// <param name="primaryKey">The primary key of the hotkey. This controls the state.</param>
    /// <param name="modifiers">List of key modifiers that are required to be held down in addition to the primary key. Such as Ctrl, Shift, Alt.</param>
    public static void RegisterHotkey(string key, KeyboardKey primaryKey, params KeyboardKey[] modifiers) {
        Hotkeys.Add(key, new Hotkey(key, primaryKey, modifiers));
    }

    /// <summary>
    /// Registeres a new hotkey with the given name, mouse button, and modifiers.
    /// </summary>
    /// <param name="key">The unique hotkey identifier</param>
    /// <param name="button">The primary mouse button of the hotkey. This controls the state.</param>
    /// <param name="modifiers">List of key modifiers that are required to be held down in addition to the primary key. Such as Ctrl, Shift, Alt.</param>
    public static void RegisterHotkey(string key, MouseButton button, params KeyboardKey[] modifiers) {
        Hotkeys.Add(key, new Hotkey(key, button, modifiers));
    }

    /// <summary>
    /// Unregisteres the hotkey with the given name.
    /// </summary>
    /// <param name="key"></param>
    public static void UnregisterHotkey(string key) {
        Hotkeys.Remove(key);
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

    public static string GetHotkeyDisplayString(string hotkey) {
        if (!Hotkeys.TryGetValue(hotkey, out Hotkey? hotkeyObj))
            return string.Empty;

        return hotkeyObj.ToDisplayString();
    }
}

/// <summary>
/// Helper class to keep track of hotkey data.
/// </summary>
internal class Hotkey {
    public string Key { get; }
    public object PrimaryKey { get; }
    public IReadOnlyList<KeyboardKey> Modifiers { get; }
    public eInteractionState State { get; set; }

    public Hotkey(string name, object primaryKey, params KeyboardKey[] modifiers) {
        Key = name;
        PrimaryKey = primaryKey;
        Modifiers = modifiers;
        State = eInteractionState.Up;
    }

    public string ToDisplayString() {
        string pKeyStr = PrimaryKey switch {
            KeyboardKey key => KeyboardKeyToString(key),
            MouseButton button => MouseButtonToString(button),
            _ => throw new NotImplementedException()
        };

        string modifiersStr = Modifiers.Count > 0 ? (" + " + string.Join(" + ", Modifiers.Select(KeyboardKeyToString))) : "";

        return $"{pKeyStr}{modifiersStr}";
    }

    public override string ToString() => $"{Key} {PrimaryKey} {State}";

    private static string KeyboardKeyToString(KeyboardKey key) {
        return key.ToString().Replace("KEY_", "");
    }

    private static string MouseButtonToString(MouseButton button) {
        return button switch {
            MouseButton.MOUSE_BUTTON_LEFT => "LMB",
            MouseButton.MOUSE_BUTTON_RIGHT => "RMB",
            MouseButton.MOUSE_BUTTON_MIDDLE => "MMB",
            _ => button.ToString().Replace("MOUSE_BUTTON_", "")
        };
    }
}
