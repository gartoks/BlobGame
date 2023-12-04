using BlobGame.App;
using BlobGame.Game.Tutorial.Stages;
using BlobGame.Util;
using OpenTK.Mathematics;
using OpenTK.Windowing.GraphicsLibraryFramework;
using SimpleGL.Graphics.Rendering;

namespace BlobGame.Game.Tutorial;
internal sealed class TutorialDisplay {
    private const float HOLD_TIME = 0.5f;

    private int CurrentStageIndex { get; set; }
    private IReadOnlyList<TutorialStage> Stages { get; }
    private TutorialStage? CurrentStage => CurrentStageIndex < 0 || CurrentStageIndex >= Stages.Count ? null : Stages[CurrentStageIndex];

    public bool IsFinished => CurrentStageIndex >= Stages.Count;

    private float HoldTime { get; set; }
    private bool AdvanceStage { get; set; }

    //private TextureResource OverlayTexture { get; set; }

    public TutorialDisplay() {
        Stages = new TutorialStage[] {
            new TutorialStage1(),
            new TutorialStage2(),
            new TutorialStage3(),
            new TutorialStage4(),
            new TutorialStage5(),
            new TutorialStage6(),
            new TutorialStage7(),
        };

        CurrentStageIndex = 0;

        HoldTime = 0;
        AdvanceStage = false;
    }

    internal void Load() {
        //OverlayTexture = ResourceManager.TextureLoader.Get("tutorial_stage_1_overlay");

        CurrentStage?.Load();
    }

    internal void Draw() {
        if (IsFinished)
            return;

        Primitives.DrawRectangle(
            Vector2.Zero,
            new Vector2(GameApplication.PROJECTION_WIDTH, GameApplication.PROJECTION_HEIGHT),
            Vector2.Zero,
            0,
            5,
            Color4.White.ChangeAlpha(64));

        if (!CurrentStage!.IsFadeInFinished)
            CurrentStage.DrawFadeIn();
        else if (!AdvanceStage)
            CurrentStage.Draw();
        else if (!CurrentStage.IsFadeOutFinished)
            CurrentStage.DrawFadeOut();
    }

    internal void Update(float dT) {
        if (CurrentStage != null &&
            CurrentStage.IsFadeInFinished &&
            !AdvanceStage &&
            Input.IsMouseButtonDown(MouseButton.Left) &&
            !Input.WasMouseHandled[MouseButton.Left])
            HoldTime += dT;

        if (!AdvanceStage && HoldTime >= HOLD_TIME) {
            Input.WasMouseHandled[MouseButton.Left] = true;

            HoldTime = 0;
            AdvanceStage = true;
        }

        if (CurrentStage != null && CurrentStage.IsFadeOutFinished && AdvanceStage) {
            CurrentStage?.Unload();
            CurrentStageIndex++;
            CurrentStage?.Load();
            AdvanceStage = false;
        }
    }

    internal void Unload() {
        CurrentStage?.Unload();
    }
}

