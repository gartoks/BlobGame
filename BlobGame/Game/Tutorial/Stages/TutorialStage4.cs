﻿using BlobGame.Audio;
using BlobGame.Drawing;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Tutorial.Stages;
internal class TutorialStage4 : TutorialStage {
    private TextureResource SpeechbubbleTexture { get; set; }

    private AnimatedTexture AnimatedSpeechbubble { get; set; }
    private AnimatedTexture AnimatedAvatarFadeOut { get; set; }
    private AnimatedTexture AnimatedPointer { get; set; }
    private Vector2 PointerAnimationDirection { get; }

    internal override bool IsFadeInFinished => true;
    internal override bool IsFadeOutFinished => AnimatedAvatarFadeOut != null && AnimatedAvatarFadeOut.IsFinished;

    private bool PlayedSound { get; set; }

    public TutorialStage4() {
        PlayedSound = false;

        PointerAnimationDirection = new Vector2(MathF.Cos(MathF.PI / 2f + 0 * RayMath.DEG2RAD), MathF.Sin(MathF.PI / 2f + 0 * RayMath.DEG2RAD));
    }

    internal override void Load() {
        base.Load();
        SpeechbubbleTexture = ResourceManager.TextureLoader.Get("speechbubble");

        AnimatedSpeechbubble = new AnimatedTexture(
            SpeechbubbleTexture,
            2,
            new Vector2(Application.BASE_WIDTH / 2 + 20, Application.BASE_HEIGHT / 2 + 70),
            new Vector2(0.5f, 0.5f)) {
            ScaleAnimator = t => Vector2.One + new Vector2(0.05f, 0.05f) * GetSpeechbubbleScaleT(t),
            RotationAnimator = t => MathF.PI / 128 * GetSpeechbubbleRotationT(t)
        };

        AnimatedAvatarFadeOut = new AnimatedTexture(
            AvatarTexture,
            0.65f,
            new Vector2(-100, Application.BASE_HEIGHT + AvatarTexture.Resource.height / 2),
            new Vector2(0, 1)) {
            PositionAnimator = t => new Vector2(0, GetAvatarPositionT(t) * AvatarTexture.Resource.height / 2f)
        };

        AnimatedPointer = new AnimatedTexture(
            PointerTexture,
            0.5f,
            new Vector2(0.814f * Application.BASE_WIDTH, 0.170f * Application.BASE_HEIGHT),
            Vector2.One / 2f,
            Vector2.One / 2f,
            -180f * RayMath.DEG2RAD) {
            PositionAnimator = t => PointerAnimationDirection * 10 * -MathF.Sin(MathF.Tau * t)
        };
    }

    internal override void Unload() {
        base.Unload();
    }

    internal override void DrawFadeIn() {
    }

    internal override void Draw() {
        if (!PlayedSound) {
            AudioManager.PlaySound("tutorial_4");
            PlayedSound = true;
        }

        AvatarTexture.Draw(new Vector2(-100, Application.BASE_HEIGHT - AvatarTexture.Resource.height / 2));

        if (AnimatedPointer.IsReady)
            AnimatedPointer.Start();

        AnimatedPointer.Draw();

        if (AnimatedPointer.IsFinished)
            AnimatedPointer.Start();

        DrawSpeechBubble();

        Renderer.GuiFont.Draw(
            "Pieces will combine like this!",
            50,
            ResourceManager.ColorLoader.Get("dark_accent"),
            new Vector2(600, Application.BASE_HEIGHT / 2 - 100));

        DrawLMBHint(750);
    }

    internal override void DrawFadeOut() {
        if (AnimatedAvatarFadeOut.IsReady)
            AnimatedAvatarFadeOut.Start();
        AnimatedAvatarFadeOut.Draw();
    }

    private void DrawSpeechBubble() {
        if (AnimatedSpeechbubble.IsReady)
            AnimatedSpeechbubble.Start();

        AnimatedSpeechbubble.Draw();

        if (AnimatedSpeechbubble.IsFinished)
            AnimatedSpeechbubble.Start();
    }

    private float GetAvatarPositionT(float t) {
        float tmp = 1.3f * t - 1;
        return 1.1f * (-(tmp * tmp) + 1);
    }

    private float GetSpeechbubbleScaleT(float t) {
        return 0.5f * (MathF.Sin(MathF.Tau * t - MathF.PI / 2) + 1);
    }

    private float GetSpeechbubbleRotationT(float t) {
        return -MathF.Sin(MathF.Tau * t);
    }

}
