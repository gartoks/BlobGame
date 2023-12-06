using BlobGame.App;
using BlobGame.Game.GameModes;
using Raylib_CsLo;

namespace BlobGame.Game.GameControllers;
internal sealed class KeyboardController : IGameController {
    private const float CURSOR_SPEED = 0.75f;

    private float _CurrentT { get; set; }
    private float CurrentT {
        get => _CurrentT;
        set => _CurrentT = Math.Clamp(value, 0, 1);
    }

    public KeyboardController() {
        CurrentT = 0;
    }

    public void Load() {
        Input.RegisterHotkey("cursor_left", KeyboardKey.KEY_A);
        Input.RegisterHotkey("cursor_right", KeyboardKey.KEY_D);
        Input.RegisterHotkey("drop_piece", KeyboardKey.KEY_SPACE);
        Input.RegisterHotkey("hold_piece", KeyboardKey.KEY_TAB);
    }

    public float GetCurrentT() {
        return CurrentT;
    }

    public bool SpawnBlob(IGameMode simulation, out float t) {
        t = GetCurrentT();

        if (!simulation.CanSpawnBlob)
            return false;

        return Input.IsHotkeyActive("drop_piece");
    }

    public bool HoldBlob() {
        return Input.IsHotkeyActive("hold_piece");
    }

    public void Update(float dT, IGameMode simulation) {
        if (Input.IsHotkeyDown("cursor_left"))
            CurrentT -= dT * CURSOR_SPEED;
        if (Input.IsHotkeyDown("cursor_right"))
            CurrentT += dT * CURSOR_SPEED;
    }

    public void Close() {
        Input.UnregisterHotkey("cursor_left");
        Input.UnregisterHotkey("cursor_right");
        Input.UnregisterHotkey("drop_piece");
    }
}
