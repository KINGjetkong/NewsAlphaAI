// src/components/DeephavenSyncPanel.jsx
import React, { useState, useEffect, useRef } from 'react'
import { Activity, Zap, RefreshCw, Radio, Layers, TrendingUp, TrendingDown } from 'lucide-react'
import clsx from 'clsx'
import { api, wsUrl, fmt } from '../utils/api.js'

export default function DeephavenSyncPanel({ ticker }) {
  const [ticks, setTicks] = useState([])
  const [velocity, setVelocity] = useState(null)
  const [clusters, setClusters] = useState([])
  const [regime, setRegime] = useState(null)
  const [connected, setConnected] = useState(false)
  const [fetchingFeeds, setFetchingFeeds] = useState(false)
  const wsRef = useRef(null)

  // Fetch initial REST snapshot
  const loadSnapshot = async () => {
    if (!ticker) return
    try {
      const data = await api.get(`/api/deephaven/${ticker}/snapshot`)
      if (data) {
        if (data.synchronized_ticks) setTicks(data.synchronized_ticks)
        if (data.sentiment_velocity) setVelocity(data.sentiment_velocity)
        if (data.cross_modal_clusters) setClusters(data.cross_modal_clusters)
        if (data.market_regime) setRegime(data.market_regime)
      }
    } catch (err) {
      console.warn('Deephaven snapshot load error:', err)
    }
  }

  // Trigger concurrent multi-source ingestion
  const refreshAlternativeFeeds = async () => {
    if (!ticker || fetchingFeeds) return
    setFetchingFeeds(true)
    try {
      const data = await api.get(`/api/sentiment/multi-source/${ticker}`)
      if (data) {
        if (data.sentiment_velocity) setVelocity(data.sentiment_velocity)
        if (data.cross_modal_clusters) setClusters(data.cross_modal_clusters)
      }
      await loadSnapshot()
    } catch (err) {
      console.warn('Alternative feeds refresh error:', err)
    } finally {
      setFetchingFeeds(false)
    }
  }

  // Connect WebSocket for real-time Deephaven streaming
  useEffect(() => {
    if (!ticker) return
    loadSnapshot()

    const url = wsUrl(`/ws/deephaven/${ticker}`)
    let ws = null
    try {
      ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
      }

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data)
          if (msg.type === 'deephaven_init') {
            if (msg.synchronized_ticks) setTicks(msg.synchronized_ticks)
            if (msg.sentiment_velocity) setVelocity(msg.sentiment_velocity)
            if (msg.cross_modal_clusters) setClusters(msg.cross_modal_clusters)
          } else if (msg.type === 'asof_tick' && msg.data) {
            setTicks((prev) => [msg.data, ...prev.slice(0, 49)])
          } else if (msg.type === 'unified_event') {
            // New event arrived, update snapshot
            loadSnapshot()
          }
        } catch (err) {
          console.warn('Deephaven WS message parse error:', err)
        }
      }

      ws.onclose = () => setConnected(false)
      ws.onerror = () => setConnected(false)
    } catch (err) {
      console.warn('Deephaven WS connection error:', err)
    }

    return () => {
      if (ws) ws.close()
    }
  }, [ticker])

  const velScore = velocity?.sentiment_velocity ?? 0
  const isShock = velocity?.shock_detected

  return (
    <div className="card p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-bg-border pb-3">
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-accent-blue" />
          <span className="stat-label text-text-primary">
            DEEPHAVEN AS-OF TICK & SENTIMENT SYNC
          </span>
          <span
            className={clsx(
              'px-2 py-0.5 rounded-full text-[10px] font-mono flex items-center space-x-1',
              connected ? 'bg-accent-green/20 text-accent-green' : 'bg-accent-yellow/20 text-accent-yellow'
            )}
          >
            <Radio className="w-2.5 h-2.5 animate-pulse inline mr-1" />
            {connected ? 'LIVE AS-OF STREAM' : 'BUFFERED SYNC'}
          </span>
        </div>

        <button
          onClick={refreshAlternativeFeeds}
          disabled={fetchingFeeds}
          className="btn-ghost text-xs flex items-center space-x-1"
        >
          <RefreshCw className={clsx('w-3.5 h-3.5', fetchingFeeds && 'animate-spin')} />
          <span>{fetchingFeeds ? 'Ingesting Feeds…' : 'Sync All Feeds'}</span>
        </button>
      </div>

      {/* Top Indicators: Velocity & HMM Regime */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Sentiment Velocity */}
        <div className="bg-bg-secondary p-3 rounded-lg border border-bg-border">
          <div className="flex justify-between items-center text-[11px] text-text-muted mb-1">
            <span>SENTIMENT VELOCITY d(S)/dt</span>
            <Zap className={clsx('w-3.5 h-3.5', isShock ? 'text-accent-red animate-bounce' : 'text-accent-blue')} />
          </div>
          <div className="flex items-baseline space-x-2">
            <span
              className={clsx(
                'text-lg font-bold font-mono',
                velScore > 0.1 ? 'text-accent-green' : velScore < -0.1 ? 'text-accent-red' : 'text-text-primary'
              )}
            >
              {velScore > 0 ? `+${velScore.toFixed(3)}` : velScore.toFixed(3)}
            </span>
            <span className="text-[10px] text-text-muted">15m vs 1h</span>
          </div>
          <div className="text-[10px] text-text-muted mt-1 flex justify-between">
            <span>Active items (15m): {velocity?.event_count_15m ?? 0}</span>
            <span>Dominant: {velocity?.dominant_modality?.toUpperCase() ?? 'NONE'}</span>
          </div>
        </div>

        {/* HMM Market Regime */}
        <div className="bg-bg-secondary p-3 rounded-lg border border-bg-border">
          <div className="flex justify-between items-center text-[11px] text-text-muted mb-1">
            <span>HMM MARKET REGIME</span>
            <Layers className="w-3.5 h-3.5 text-accent-cyan" />
          </div>
          <div className="text-sm font-semibold font-mono text-accent-cyan truncate">
            {regime?.regime ?? 'CALCULATING…'}
          </div>
          <div className="text-[10px] text-text-muted mt-1 flex justify-between">
            <span>Confidence: {regime ? `${(regime.confidence * 100).toFixed(0)}%` : '—'}</span>
            <span className={clsx(
              'font-semibold',
              (regime?.tilt_multiplier ?? 0) > 0 ? 'text-accent-green' : (regime?.tilt_multiplier ?? 0) < 0 ? 'text-accent-red' : 'text-text-secondary'
            )}>
              Tilt: {regime?.tilt_multiplier != null ? `${regime.tilt_multiplier > 0 ? '+' : ''}${regime.tilt_multiplier}` : '0.0'}
            </span>
          </div>
        </div>

        {/* Cross-Modal Clusters */}
        <div className="bg-bg-secondary p-3 rounded-lg border border-bg-border">
          <div className="flex justify-between items-center text-[11px] text-text-muted mb-1">
            <span>CROSS-MODAL CONFLUENCE</span>
            {clusters.length > 0 ? (
              <TrendingUp className="w-3.5 h-3.5 text-accent-green" />
            ) : (
              <TrendingDown className="w-3.5 h-3.5 text-text-muted" />
            )}
          </div>
          <div className="text-sm font-semibold font-mono text-text-primary">
            {clusters.length} Active {clusters.length === 1 ? 'Cluster' : 'Clusters'}
          </div>
          <div className="text-[10px] text-text-muted mt-1 truncate">
            {clusters[0]?.headline_summary || 'No multi-source clusters detected'}
          </div>
        </div>
      </div>

      {/* Streaming As-Of Joined Ticks Table */}
      <div className="space-y-2">
        <div className="flex justify-between items-center text-xs text-text-muted">
          <span>REAL-TIME AS-OF JOINED TICKS (ZERO LOOKAHEAD)</span>
          <span className="font-mono text-[10px]">{ticks.length} prints in memory</span>
        </div>

        <div className="overflow-x-auto max-h-64 border border-bg-border rounded-lg">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-bg-secondary text-text-muted sticky top-0">
              <tr>
                <th className="p-2">Time</th>
                <th className="p-2">Price</th>
                <th className="p-2">Size</th>
                <th className="p-2">Prior Aligned Event</th>
                <th className="p-2">FinBERT</th>
                <th className="p-2">Lead</th>
                <th className="p-2">Displacement</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-bg-border/40">
              {ticks.length === 0 ? (
                <tr>
                  <td colSpan={7} className="p-4 text-center text-text-muted">
                    Waiting for incoming market ticks or alternative events…
                  </td>
                </tr>
              ) : (
                ticks.map((t, idx) => {
                  const sent = t.prior_sentiment_score ?? 0
                  const isBull = sent > 0.15
                  const isBear = sent < -0.15
                  const disp = t.post_shock_displacement_bps ?? 0

                  return (
                    <tr key={t.tick_id || idx} className="hover:bg-bg-secondary/40 transition-colors">
                      <td className="p-2 text-text-muted">
                        {t.timestamp ? new Date(t.timestamp).toLocaleTimeString() : '—'}
                      </td>
                      <td className="p-2 font-semibold text-text-primary">
                        {fmt.price(t.price)}
                      </td>
                      <td className="p-2 text-text-muted">{t.size || 100}</td>
                      <td className="p-2 max-w-[220px] truncate" title={t.prior_headline || ''}>
                        {t.prior_source && (
                          <span className="mr-1.5 px-1 py-0.2 rounded text-[9px] bg-accent-blue/15 text-accent-blue">
                            {t.prior_source.toUpperCase()}
                          </span>
                        )}
                        <span className="text-text-secondary">{t.prior_headline || 'Base State'}</span>
                      </td>
                      <td className="p-2">
                        <span
                          className={clsx(
                            'px-1.5 py-0.5 rounded text-[10px] font-semibold',
                            isBull
                              ? 'bg-accent-green/15 text-accent-green'
                              : isBear
                              ? 'bg-accent-red/15 text-accent-red'
                              : 'bg-text-muted/15 text-text-secondary'
                          )}
                        >
                          {sent > 0 ? `+${sent.toFixed(2)}` : sent.toFixed(2)}
                        </span>
                      </td>
                      <td className="p-2 text-text-muted">
                        {t.sentiment_lead_time_seconds != null
                          ? `${t.sentiment_lead_time_seconds.toFixed(1)}s`
                          : '0s'}
                      </td>
                      <td
                        className={clsx(
                          'p-2 font-semibold',
                          disp > 0 ? 'text-accent-green' : disp < 0 ? 'text-accent-red' : 'text-text-muted'
                        )}
                      >
                        {disp > 0 ? `+${disp} bps` : `${disp} bps`}
                      </td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
