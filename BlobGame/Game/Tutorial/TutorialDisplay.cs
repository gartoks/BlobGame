﻿using BlobGame.App;
using BlobGame.Game.Tutorial.Stages;
using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Tutorial;
internal sealed class TutorialDisplay {
    private const float HOLD_TIME = 0.1f;

    private int CurrentStageIndex { get; set; }
    private IReadOnlyList<TutorialStage> Stages { get; }
    private TutorialStage? CurrentStage => CurrentStageIndex < 0 || CurrentStageIndex >= Stages.Count ? null : Stages[CurrentStageIndex];

    public bool IsFinished => CurrentStageIndex >= Stages.Count;

    private float HoldTime { get; set; }
    private bool AdvanceStage { get; set; }

    private TextureResource OverlayTexture { get; set; }

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
        OverlayTexture = ResourceManager.GetTexture("tutorial_stage_1_overlay");

        CurrentStage?.Load();
    }

    internal void Draw() {
        if (IsFinished)
            return;

        OverlayTexture.Draw(Vector2.Zero);

        if (!CurrentStage!.IsFadeInFinished)
            CurrentStage.DrawFadeIn();
        else if (!AdvanceStage)
            CurrentStage.Draw();
        else if (!CurrentStage.IsFadeOutFinished)
            CurrentStage.DrawFadeOut();
    }

    internal void Update(float dT) {
        if (CurrentStage != null && CurrentStage.IsFadeInFinished && !AdvanceStage && Input.IsMouseButtonDown(MouseButton.MOUSE_BUTTON_LEFT))
            HoldTime += dT;

        if (!AdvanceStage && HoldTime >= HOLD_TIME) {
            Input.WasMouseHandled[MouseButton.MOUSE_BUTTON_LEFT] = true;

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

