package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
)

func TestTranscriptCorrectionRejectsEmptyInput(t *testing.T) {
	gin.SetMode(gin.TestMode)
	handler := newTranscriptCorrectionHandler(&config.Config{}, http.DefaultClient)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/transcript-corrections/gemma", strings.NewReader(`{"input":""}`))
	ctx.Request.Header.Set("Content-Type", "application/json")

	handler(ctx)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
}

func TestTranscriptCorrectionRejectsTrailingJSON(t *testing.T) {
	handler := newTranscriptCorrectionHandler(&config.Config{}, http.DefaultClient)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/transcript-corrections/gemma", strings.NewReader(`{"input":"文字"}{"input":"second"}`))
	handler(ctx)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
}

func TestTranscriptCorrectionRejectsOversizeInput(t *testing.T) {
	handler := newTranscriptCorrectionHandler(&config.Config{}, http.DefaultClient)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	body := `{"input":"` + strings.Repeat("x", transcriptMaxInputBytes+1) + `"}`
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/transcript-corrections/gemma", strings.NewReader(body))
	handler(ctx)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
}

func TestTranscriptCorrectionHTTPClientDoesNotFollowRedirects(t *testing.T) {
	targetRequests := 0
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		targetRequests++
	}))
	defer target.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()
	req, _ := http.NewRequest(http.MethodPost, redirect.URL, strings.NewReader("payload"))
	req.Header.Set("CF-Access-Client-Secret", "must-not-leak")
	response, err := newTranscriptCorrectionHTTPClient().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusTemporaryRedirect || targetRequests != 0 {
		t.Fatalf("status=%d target_requests=%d", response.StatusCode, targetRequests)
	}
}

func TestTranscriptCorrectionHTTPClientFollowsSameOriginRedirects(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path == "/start" {
			http.Redirect(w, r, "/finish", http.StatusTemporaryRedirect)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	req, _ := http.NewRequest(http.MethodPost, server.URL+"/start", strings.NewReader("payload"))
	response, err := newTranscriptCorrectionHTTPClient().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK || requests != 2 {
		t.Fatalf("status=%d requests=%d", response.StatusCode, requests)
	}
}

func TestTranscriptCorrectionUsesFixedNativeLMStudioRequest(t *testing.T) {
	var upstream map[string]any
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/chat" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("X-CF-Test"); got != "server-secret" {
			t.Fatalf("upstream header = %q", got)
		}
		if err := json.NewDecoder(r.Body).Decode(&upstream); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"output":[{"type":"message","content":"校正文字"}]}`))
	}))
	defer upstreamServer.Close()

	cfg := &config.Config{OpenAICompatibility: []config.OpenAICompatibility{{
		Name:    "LM Studio MacBook",
		BaseURL: upstreamServer.URL + "/v1",
		Headers: map[string]string{"X-CF-Test": "server-secret"},
		Models:  []config.OpenAICompatibilityModel{{Name: "actual-gemma", Alias: "gemma-4-26b"}},
	}}}
	handler := newTranscriptCorrectionHandler(cfg, upstreamServer.Client())
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/transcript-corrections/gemma", strings.NewReader(`{"input":"原始文字","max_output_tokens":900}`))
	ctx.Request.Header.Set("Content-Type", "application/json")

	handler(ctx)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", recorder.Code, recorder.Body.String())
	}
	var response map[string]string
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response["text"] != "校正文字" || response["model"] != "vps-native:gemma-4-26b" {
		t.Fatalf("response = %#v", response)
	}
	if upstream["model"] != "actual-gemma" || upstream["reasoning"] != "off" || upstream["store"] != false || upstream["stream"] != false {
		t.Fatalf("unsafe upstream payload: %#v", upstream)
	}
	if upstream["temperature"] != float64(0) || upstream["max_output_tokens"] != float64(900) {
		t.Fatalf("unexpected limits: %#v", upstream)
	}
}

func TestTranscriptCorrectionRedactsUpstreamFailure(t *testing.T) {
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "server-secret sensitive body", http.StatusBadGateway)
	}))
	defer upstreamServer.Close()
	cfg := &config.Config{OpenAICompatibility: []config.OpenAICompatibility{{
		Name: "LM Studio MacBook", BaseURL: upstreamServer.URL + "/v1",
		Models: []config.OpenAICompatibilityModel{{Name: "actual-gemma", Alias: "gemma-4-26b"}},
	}}}
	handler := newTranscriptCorrectionHandler(cfg, upstreamServer.Client())
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/transcript-corrections/gemma", strings.NewReader(`{"input":"文字"}`))

	handler(ctx)

	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status = %d", recorder.Code)
	}
	if strings.Contains(recorder.Body.String(), "server-secret") || strings.Contains(recorder.Body.String(), "sensitive body") {
		t.Fatalf("response leaked upstream details: %s", recorder.Body.String())
	}
}
