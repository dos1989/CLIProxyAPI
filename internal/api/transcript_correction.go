package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
)

const (
	transcriptGemmaAlias   = "gemma-4-26b"
	transcriptGemmaRouteID = "vps-native:gemma-4-26b"
	transcriptProviderName = "LM Studio MacBook"
)

type transcriptCorrectionRequest struct {
	Input           string `json:"input"`
	MaxOutputTokens int    `json:"max_output_tokens,omitempty"`
}

type nativeChatResponse struct {
	Output []struct {
		Type    string `json:"type"`
		Content string `json:"content"`
	} `json:"output"`
}

func newTranscriptCorrectionHandler(cfg *config.Config, client *http.Client) gin.HandlerFunc {
	return func(c *gin.Context) {
		var input transcriptCorrectionRequest
		decoder := json.NewDecoder(c.Request.Body)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil || strings.TrimSpace(input.Input) == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid transcript correction request"})
			return
		}
		if input.MaxOutputTokens == 0 {
			input.MaxOutputTokens = 1200
		}
		if input.MaxOutputTokens < 1 || input.MaxOutputTokens > 4096 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid transcript correction request"})
			return
		}

		provider, model, err := transcriptGemmaProvider(cfg)
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "transcript correction route unavailable"})
			return
		}
		payload := map[string]any{
			"model":             model,
			"input":             input.Input,
			"reasoning":         "off",
			"store":             false,
			"stream":            false,
			"temperature":       0,
			"max_output_tokens": input.MaxOutputTokens,
		}
		body, _ := json.Marshal(payload)
		request, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, nativeChatURL(provider.BaseURL), bytes.NewReader(body))
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": "transcript correction upstream failed"})
			return
		}
		request.Header.Set("Content-Type", "application/json")
		for name, value := range provider.Headers {
			request.Header.Set(name, value)
		}
		if len(provider.APIKeyEntries) > 0 && strings.TrimSpace(provider.APIKeyEntries[0].APIKey) != "" {
			request.Header.Set("Authorization", "Bearer "+strings.TrimSpace(provider.APIKeyEntries[0].APIKey))
		}

		response, err := client.Do(request)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": "transcript correction upstream failed"})
			return
		}
		defer response.Body.Close()
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			c.JSON(http.StatusBadGateway, gin.H{"error": "transcript correction upstream failed"})
			return
		}
		var native nativeChatResponse
		if err := json.NewDecoder(response.Body).Decode(&native); err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": "transcript correction upstream failed"})
			return
		}
		text := ""
		for _, item := range native.Output {
			if item.Type == "message" && strings.TrimSpace(item.Content) != "" {
				text = strings.TrimSpace(item.Content)
				break
			}
		}
		if text == "" {
			c.JSON(http.StatusBadGateway, gin.H{"error": "transcript correction upstream failed"})
			return
		}
		c.JSON(http.StatusOK, gin.H{"text": text, "model": transcriptGemmaRouteID})
	}
}

func transcriptGemmaProvider(cfg *config.Config) (*config.OpenAICompatibility, string, error) {
	if cfg == nil {
		return nil, "", errors.New("missing configuration")
	}
	for i := range cfg.OpenAICompatibility {
		provider := &cfg.OpenAICompatibility[i]
		if provider.Disabled || provider.Name != transcriptProviderName {
			continue
		}
		for _, model := range provider.Models {
			if model.Alias == transcriptGemmaAlias && strings.TrimSpace(model.Name) != "" {
				return provider, model.Name, nil
			}
		}
	}
	return nil, "", errors.New("missing transcript Gemma provider")
}

func nativeChatURL(baseURL string) string {
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return ""
	}
	parsed.Path = strings.TrimSuffix(strings.TrimSuffix(parsed.Path, "/"), "/v1") + "/api/v1/chat"
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String()
}
