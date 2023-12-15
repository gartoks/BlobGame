using BlobGame.Audio;
using BlobGame.Drawing;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Tutorial;
internal sealed class TutorialStage {
    //const int AVATAR_WIDTH = 944;
    //const int AVATAR_HEIGHT = 1432;
    const int AVATAR_WIDTH = 768;
    const int AVATAR_HEIGHT = 768;
    const float FONT_SIZE = 50;

    private TutorialDisplay Tutorial { get; }
    private int StageIndex { get; }
    public string Text { get; }
    public Vector2 PointerPos { get; }
    public float PointerRot { get; }
    public float AvatarX { get; }
    public Vector2 SpeechBubblePos { get; }
    public int SpeechFrames { get; }
    public int OverlayIndex { get; }

    internal bool IsFadeInFinished => !HasFadeIn || AnimatedAvatarFadeIn!.IsFinished;
    internal bool IsFadeOutFinished => !HasFadeOut || AnimatedAvatarFadeOut!.IsFinished;

    private TextureResource LMBTexture { get; set; }
    private TextureResource PointerTexture { get; set; }
    private NPatchTextureResource SpeechbubbleTexture { get; set; }
    private TextureResource NameTagTexture { get; set; }

    private TextureResource AvatarIdleTexture { get; set; }
    private TextureResource AvatarBlink0Texture { get; set; }
    private TextureResource AvatarBlink1Texture { get; set; }
    private TextureResource AvatarTalk0Texture { get; set; }
    private TextureResource AvatarTalk1Texture { get; set; }
    private TextureResource AvatarTalk2Texture { get; set; }
    private TextureResource? AvatarOverlayTexture { get; set; }

    private AnimatedTexture? AnimatedAvatarFadeIn { get; set; }
    private AnimatedTexture? AnimatedAvatarFadeOut { get; set; }
    private AnimatedTexture AnimatedPointer { get; set; }
    private AnimatedTexture AnimatedSpeechbubble { get; set; }
    private FrameAnimator AvatarAnimator { get; set; }

    private bool HasFadeIn => StageIndex == 0 || AvatarX != Tutorial.Stages[StageIndex - 1].AvatarX;
    private bool HasFadeOut => StageIndex == Tutorial.Stages.Count - 1 || AvatarX != Tutorial.Stages[StageIndex + 1].AvatarX;

    private Vector2 PointerAnimationDirection { get; }
    private bool PlayedSound { get; set; }
    private int TalkFrames { get; set; }

    public TutorialStage(
        TutorialDisplay tutorial, int stageIndex, string text,
        Vector2 pointerPos, float pointerRot, float avatarX,
        Vector2 speechBubblePos, int speechFrames, int overlayIndex) {
        Tutorial = tutorial;
        StageIndex = stageIndex;
        Text = text;
        PointerPos = pointerPos;
        PointerRot = pointerRot;
        AvatarX = avatarX;
        SpeechBubblePos = speechBubblePos;
        SpeechFrames = speechFrames;
        OverlayIndex = overlayIndex;

        PointerAnimationDirection = new Vector2(MathF.Cos(MathF.PI / 2f + PointerRot * RayMath.DEG2RAD), MathF.Sin(MathF.PI / 2 + PointerRot * RayMath.DEG2RAD));

        TalkFrames = 0;
        AvatarAnimator = new FrameAnimator(1f / 24f);
    }

    internal void Load() {
        LMBTexture = ResourceManager.TextureLoader.Get("lmb");
        PointerTexture = ResourceManager.TextureLoader.Get("pointer");
        SpeechbubbleTexture = ResourceManager.NPatchTextureLoader.Get("speechbubble");
        ResourceManager.SoundLoader.Load($"{Tutorial.GameModeKey}_tutorial_{StageIndex}");
        NameTagTexture = ResourceManager.TextureLoader.Get("nametag");

        AvatarIdleTexture = ResourceManager.TextureLoader.Get("avatar_idle");
        AvatarBlink0Texture = ResourceManager.TextureLoader.Get("avatar_blink_0");
        AvatarBlink1Texture = ResourceManager.TextureLoader.Get("avatar_blink_1");
        AvatarTalk0Texture = ResourceManager.TextureLoader.Get("avatar_talk_0");
        AvatarTalk1Texture = ResourceManager.TextureLoader.Get("avatar_talk_1");
        AvatarTalk2Texture = ResourceManager.TextureLoader.Get("avatar_talk_2");
        AvatarTalk2Texture = ResourceManager.TextureLoader.Get("avatar_talk_2");
        AvatarOverlayTexture = OverlayIndex >= 0 ? ResourceManager.TextureLoader.Get($"avatar_overlay_{OverlayIndex}") : null;

        AvatarAnimator.AddFrameKey("idle", AvatarIdleTexture);
        AvatarAnimator.AddFrameKey("blink0", AvatarBlink0Texture);
        AvatarAnimator.AddFrameKey("blink1", AvatarBlink1Texture);
        AvatarAnimator.AddFrameKey("talk0", AvatarTalk0Texture);
        AvatarAnimator.AddFrameKey("talk1", AvatarTalk1Texture);
        AvatarAnimator.AddFrameKey("talk2", AvatarTalk2Texture);
        AvatarAnimator.AddFrameSequence("idle", 10, "idle", "idle", "idle", "idle", "idle", "idle", "idle", "idle", "idle", "idle");
        AvatarAnimator.AddFrameSequence("idle", 1, "idle", "blink1", "blink1", "blink0", "idle");
        AvatarAnimator.AddFrameSequence("talk", 1, "idle", "talk0", "talk0", "talk1", "talk1", "talk2", "talk2", "idle");
        AvatarAnimator.SetDefaultSequence("idle");

        PlayedSound = false;

        AnimatedPointer = new AnimatedTexture(
            PointerTexture,
            0.5f,
            PointerPos,
            new Vector2(256, 256),
            Vector2.One / 2f,
            PointerRot * RayMath.DEG2RAD) {
            PositionAnimator = t => PointerAnimationDirection * 10 * -MathF.Sin(MathF.Tau * t)
        };

        Vector2 textSize = Raylib.MeasureTextEx(Renderer.GuiFont.Resource, Text, FONT_SIZE, FONT_SIZE / 16f);

        AnimatedSpeechbubble = new AnimatedTexture(
            SpeechbubbleTexture,
            2,
            SpeechBubblePos,
            textSize * new Vector2(1.05f, 1.575f),
            new Vector2(0.54f, 0.5f)) {
            ScaleAnimator = t => Vector2.One + new Vector2(0.025f, 0.025f) * GetSpeechbubbleScaleT(t),
            RotationAnimator = t => MathF.PI / 256f * GetSpeechbubbleRotationT(t)
        };

        if (HasFadeIn) {
            AnimatedAvatarFadeIn = new AnimatedTexture(
                AvatarIdleTexture,
                0.65f,
                new Vector2(AvatarX, Application.BASE_HEIGHT + AVATAR_HEIGHT),
                new Vector2(AVATAR_WIDTH, AVATAR_HEIGHT),
                new Vector2(0.5f, 1)) {
                PositionAnimator = t => new Vector2(0, -GetAvatarPositionT(t) * AVATAR_HEIGHT)
            };
        }

        if (HasFadeOut) {
            AnimatedAvatarFadeOut = new AnimatedTexture(
                AvatarIdleTexture,
                0.65f,
                new Vector2(AvatarX, Application.BASE_HEIGHT + AVATAR_HEIGHT / 2),
                new Vector2(AVATAR_WIDTH, AVATAR_HEIGHT),
                new Vector2(0.5f, 1)) {
                PositionAnimator = t => new Vector2(0, GetAvatarPositionT(t) * AVATAR_HEIGHT)
            };
        }
    }

    internal void Unload() {
        if (ResourceManager.SoundLoader.IsLoaded($"{Tutorial.GameModeKey}_tutorial_{StageIndex}")) {
            ResourceManager.SoundLoader.Get($"{Tutorial.GameModeKey}_tutorial_{StageIndex}").Unload();
            AudioManager.StopSound($"{Tutorial.GameModeKey}_tutorial_{StageIndex}");
        }
    }

    internal void DrawFadeIn() {
        if (!HasFadeIn)
            return;

        if (AnimatedAvatarFadeIn!.IsReady)
            AnimatedAvatarFadeIn.Start();
        AnimatedAvatarFadeIn.Draw();
    }

    internal void DrawFadeOut() {
        if (!HasFadeOut)
            return;

        if (AnimatedAvatarFadeOut!.IsReady)
            AnimatedAvatarFadeOut.Start();
        AnimatedAvatarFadeOut.Draw();
    }

    internal void Draw(float dT) {
        if (!PlayedSound) {
            AudioManager.PlaySound($"{Tutorial.GameModeKey}_tutorial_{StageIndex}");
            PlayedSound = true;

            AvatarAnimator.StartSequence("talk");
        }

        if (TalkFrames < SpeechFrames && AvatarAnimator.IsReady) {
            AvatarAnimator.StartSequence("talk");
            TalkFrames += 1;
        }

        AvatarAnimator.Draw(dT, new Rectangle(AvatarX, Application.BASE_HEIGHT, AVATAR_WIDTH, AVATAR_HEIGHT), 0, new Vector2(0.5f, 1f), Raylib.WHITE);

        AvatarOverlayTexture?.Draw(new Rectangle(AvatarX, Application.BASE_HEIGHT, AVATAR_WIDTH, AVATAR_HEIGHT), new Vector2(0.5f, 1f), 0, Raylib.WHITE);

        if (AnimatedPointer.IsReady)
            AnimatedPointer.Start();

        if (AnimatedPointer.IsFinished)
            AnimatedPointer.Start();

        DrawSpeechBubble();

        AnimatedPointer.Draw();

        Renderer.GuiFont.Draw(
            Text,
            FONT_SIZE,
            ResourceManager.ColorLoader.Get("font_dark"),
            SpeechBubblePos,
            new Vector2(0.5f, 0.5f),
            0,
            float.MaxValue);
    }

    private void DrawSpeechBubble() {
        if (AnimatedSpeechbubble.IsReady)
            AnimatedSpeechbubble.Start();

        AnimatedSpeechbubble.Draw();

        Vector2 textSize = Raylib.MeasureTextEx(Renderer.GuiFont.Resource, Text, FONT_SIZE, FONT_SIZE / 16f);

        if (!string.IsNullOrWhiteSpace(Tutorial.AvatarName)) {
            NameTagTexture.Draw(
                new Rectangle(SpeechBubblePos.X - textSize.X / 2f, SpeechBubblePos.Y - textSize.Y / 2f - 20, 470, 100),
                new Vector2(0f, 1));

            Renderer.MainFont.Draw(
                Tutorial.AvatarName,
                FONT_SIZE,
                ResourceManager.ColorLoader.Get("font_dark"),
                SpeechBubblePos - textSize / 2f + new Vector2(60, -40),
                new Vector2(0f, 1)
                );
        }

        Vector2 advancePos = SpeechBubblePos + (textSize * new Vector2(0.8f, 1f)) / 2f;
        Raylib.DrawCircleSector(
            advancePos, 50, 180, 180 - 360 * Tutorial.AdvanceProgress,
            (int)(72 * Tutorial.AdvanceProgress) + 1, ResourceManager.ColorLoader.Get("light_accent").Resource);
        Raylib.DrawCircleV(advancePos, 37.5f, Raylib.WHITE);
        LMBTexture.Draw(
            new Rectangle(advancePos.X, advancePos.Y, 50, 75),
            new Vector2(0.5f, 0.5f), 0, ResourceManager.ColorLoader.Get("font_dark").Resource);
        Renderer.GuiFont.Draw(
            "Hold",
            FONT_SIZE / 2f,
            ResourceManager.ColorLoader.Get("font_dark"),
            advancePos - new Vector2(0, 35),
            new Vector2(0.5f, 1));

        if (AnimatedSpeechbubble.IsFinished)
            AnimatedSpeechbubble.Start();
    }

    private float GetAvatarPositionT(float t) {
        //float tmp = 1.3f * t - 1;
        return t;// 1.1f * (-(tmp * tmp) + 1);
    }

    private float GetSpeechbubbleScaleT(float t) {
        return 0.5f * (MathF.Sin(MathF.Tau * t - MathF.PI / 2) + 1);
    }

    private float GetSpeechbubbleRotationT(float t) {
        return -MathF.Sin(MathF.Tau * t);
    }
}