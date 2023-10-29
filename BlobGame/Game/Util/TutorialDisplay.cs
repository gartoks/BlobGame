using BlobGame.App;
using BlobGame.Drawing;
using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Util;
internal sealed class TutorialDisplay {
    private const float HOLD_TIME = 0.25f;

    private TextureResource AvatarTexture { get; set; }
    private TextureResource LMBTexture { get; set; }
    private TextureResource Overlay1Texture { get; set; }
    private TextureResource Speechbubble1Texture { get; set; }

    private int CurrentStage { get; set; }
    private IReadOnlyList<Action> Stages { get; }

    public bool IsFinished => CurrentStage >= Stages.Count;

    private float HoldTime { get; set; }

    public TutorialDisplay() {
        Stages = new Action[] {
            DrawStage1,
        };

        HoldTime = 0;
    }

    internal void Load() {
        AvatarTexture = ResourceManager.GetTexture("melba_avatar");
        LMBTexture = ResourceManager.GetTexture("lmb");
        Overlay1Texture = ResourceManager.GetTexture("tutorial_stage_1_overlay");
        Speechbubble1Texture = ResourceManager.GetTexture("tutorial_stage_1_speechbubble");
    }

    internal void Draw() {
        if (IsFinished)
            return;

        Stages[CurrentStage]();
    }

    internal void Update(float dT) {
        if (CurrentStage < Stages.Count && Input.IsMouseButtonDown(MouseButton.MOUSE_BUTTON_LEFT))
            HoldTime += dT;

        if (HoldTime >= HOLD_TIME) {
            Input.WasMouseHandled[MouseButton.MOUSE_BUTTON_LEFT] = true;
            CurrentStage++;
            HoldTime = 0;
        }
    }

    private void DrawStage1() {
        Overlay1Texture.Draw(Vector2.Zero);

        AvatarTexture.Draw(
            new Vector2(-100, Application.BASE_HEIGHT + AvatarTexture.Resource.height / 2),
            new Vector2(0, 1));

        Speechbubble1Texture.Draw(Vector2.Zero);

        Renderer.Font.Draw(
            "Move your mouse to decide\nwhere to drop a piece!",
            50,
            ResourceManager.GetColor("dark_accent"),
            new Vector2(600, Application.BASE_HEIGHT / 2 - 100));

        DrawLMBHint();
    }

    private void DrawLMBHint() {
        LMBTexture.Draw(
            new Vector2(750, Application.BASE_HEIGHT - 50),
            new Vector2(0, 1),
            new Vector2(0.4f, 0.4f));

        Renderer.Font.Draw(
            "Hold LMB to continue!",
            80,
            ResourceManager.GetColor("dark_accent"),
            new Vector2(875, Application.BASE_HEIGHT - 150));
    }

}
