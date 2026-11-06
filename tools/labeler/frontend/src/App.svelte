<script lang="ts">
  import { onMount } from "svelte";

  import { fetchJSON, type StageSummary } from "./lib/api";
  import { setDatasetId } from "./lib/state";
  import { workspaces } from "./stages/registry";

  let stages = $state<StageSummary[]>([]);
  let status = $state("connecting…");
  let route = $state(window.location.pathname);
  /** The health probe has reported the mounted dataset (state keys are ready). */
  let ready = $state(false);

  onMount(() => {
    const onPopState = () => (route = window.location.pathname);
    window.addEventListener("popstate", onPopState);
    void (async () => {
      try {
        const [health, list] = await Promise.all([
          fetchJSON<{ ok: boolean; version: string; dataset: string }>(
            "/api/health",
          ),
          fetchJSON<{ stages: StageSummary[] }>("/api/stages"),
        ]);
        setDatasetId(health.dataset);
        stages = list.stages;
        status = `connected · v${health.version} · ${health.dataset}`;
      } catch (err) {
        status = `backend unreachable: ${err instanceof Error ? err.message : err}`;
      } finally {
        ready = true;
      }
    })();
    return () => window.removeEventListener("popstate", onPopState);
  });

  function nav(e: MouseEvent, href: string): void {
    e.preventDefault();
    history.pushState(null, "", href);
    route = href;
  }

  const stageId = $derived(
    route === "/" ? "" : route.replace(/^\//, "").split("/")[0],
  );
  const Workspace = $derived(
    stageId && stageId in workspaces ? workspaces[stageId] : null,
  );
</script>

<div class="shell">
  <nav class="topbar">
    <a class="brand" href="/" onclick={(e) => nav(e, "/")}>labeler</a>
    {#each stages as s (s.id)}
      <a
        href="/{s.id}"
        class:on={s.id === stageId}
        onclick={(e) => nav(e, `/${s.id}`)}
      >
        {s.id}
      </a>
    {/each}
    <span class="spacer"></span>
    <span class="status">{status}</span>
  </nav>

  {#if Workspace}
    {@const W = Workspace}
    {#if ready}
      <W />
    {/if}
  {:else if route === "/"}
    <main class="picker">
      <h1>labeler</h1>
      <p class="dim">{status}</p>
      {#if stages.length > 0}
        <ul class="stages">
          {#each stages as stage (stage.id)}
            <li>
              <a href="/{stage.id}" onclick={(e) => nav(e, `/${stage.id}`)}>
                <strong>{stage.title}</strong>
                <span>{stage.description}</span>
              </a>
            </li>
          {/each}
        </ul>
      {:else}
        <p class="empty">
          No workflow stages registered yet — the annotate and review stages
          land with the shared canvas engine.
        </p>
      {/if}
    </main>
  {:else}
    <main class="picker">
      <p class="empty">Unknown route “{route}”.</p>
    </main>
  {/if}
</div>

<style>
  .shell {
    height: 100%;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .topbar {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--panel);
    border-bottom: 1px solid var(--edge);
    padding: 6px 10px;
  }
  .topbar a {
    color: var(--dim);
    text-decoration: none;
    padding: 2px 8px;
    border-radius: 5px;
  }
  .topbar a:hover {
    color: var(--text);
    background: #262a33;
  }
  .topbar a.on {
    color: #fff;
    background: var(--accent);
  }
  .topbar a.brand {
    color: var(--text);
    font-weight: 600;
  }
  .spacer {
    flex: 1;
  }
  .status {
    color: var(--dim);
    font-size: 12px;
  }
  .picker {
    max-width: 52rem;
    margin: 0 auto;
    padding: 2rem 1.5rem;
    width: 100%;
  }
  .picker h1 {
    font-size: 1.4rem;
    margin: 0 0 0.25rem;
  }
  .dim {
    color: var(--dim);
    margin: 0 0 2rem;
  }
  ul.stages {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 0.75rem;
  }
  ul.stages li a {
    display: block;
    border: 1px solid var(--edge);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    color: var(--text);
    text-decoration: none;
  }
  ul.stages li a:hover {
    background: #262a33;
  }
  ul.stages li strong {
    display: block;
  }
  ul.stages li span {
    color: var(--dim);
    font-size: 0.9rem;
  }
  .empty {
    border: 1px dashed var(--edge);
    border-radius: 6px;
    padding: 1.5rem;
    color: var(--dim);
  }
</style>
