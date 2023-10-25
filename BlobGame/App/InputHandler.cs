using Raylib_CsLo;
using System.Numerics;
using static BlobGame.App.InputHandler;

namespace BlobGame.App;
public static class InputHandler {
    public enum eInteractionState {
        Up, Pressed, Down, Released
    };

    private static Dictionary<MouseButton, eInteractionState> MouseButtonStates { get; }
    private static Dictionary<string, Hotkey> Hotkeys { get; }

    public static bool WasMouseHandled { get; set; }

    static InputHandler() {
        MouseButtonStates = new Dictionary<MouseButton, eInteractionState>();
        MouseButtonStates[MouseButton.MOUSE_BUTTON_LEFT] = eInteractionState.Up;
        MouseButtonStates[MouseButton.MOUSE_BUTTON_RIGHT] = eInteractionState.Up;
        MouseButtonStates[MouseButton.MOUSE_BUTTON_MIDDLE] = eInteractionState.Up;

        Hotkeys = new Dictionary<string, Hotkey>();
    }

    internal static void Initialize() {
    }

    internal static void Load() {
    }

    internal static void Update() {
        WasMouseHandled = false;

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
                else
                    MouseButtonStates[mbs] = eInteractionState.Up;
            }
        }

        foreach (Hotkey hotkey in Hotkeys.Values) {
            eInteractionState state = hotkey.State is eInteractionState.Released or eInteractionState.Up ? eInteractionState.Up : eInteractionState.Released;

            bool allModifiersDown = hotkey.Modifiers.All(Raylib.IsKeyDown);
            if (allModifiersDown) {
                if (Raylib.IsKeyPressed(hotkey.PrimaryKey))
                    state = eInteractionState.Pressed;
                else if (Raylib.IsKeyDown(hotkey.PrimaryKey))
                    state = eInteractionState.Down;
                else if (Raylib.IsKeyReleased(hotkey.PrimaryKey))
                    state = eInteractionState.Released;
            }
            hotkey.State = state;
        }
    }

    public static void RegisterHotkey(string name, KeyboardKey primaryKey, params KeyboardKey[] modifiers) {
        Hotkeys.Add(name, new Hotkey(name, primaryKey, modifiers));
    }

    public static eInteractionState GetMouseButtonState(MouseButton button) {
        return MouseButtonStates[button];
    }

    public static bool IsHotkeyActive(string hotkey) {
        return Hotkeys[hotkey].State is eInteractionState.Released;
    }

    public static bool IsHotkeyDown(string hotkey) {
        return Hotkeys[hotkey].State is eInteractionState.Pressed or eInteractionState.Down;
    }

    public static bool IsMouseButtonActive(MouseButton button) {
        return !WasMouseHandled && MouseButtonStates[button] == eInteractionState.Released;
    }
}

internal class Hotkey {
    public string Name { get; }
    public KeyboardKey PrimaryKey { get; }
    public IReadOnlyList<KeyboardKey> Modifiers { get; }
    public eInteractionState State { get; set; }

    public Hotkey(string name, KeyboardKey primaryKey, params KeyboardKey[] modifiers) {
        Name = name;
        PrimaryKey = primaryKey;
        Modifiers = modifiers;
        State = eInteractionState.Up;
    }

    public override string ToString() => $"{Name} {PrimaryKey} {State}";
}
