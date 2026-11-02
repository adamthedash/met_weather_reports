<script lang="ts">
  import { onMount } from "svelte";

  import { fetchJSON, type StageSummary } from "./lib/api";

  let stages = $state<StageSummary[]>([]);
  let status = $state("connecting…");

  onMount(() => {
    (async () => {
      try {
        const [health, list] = await Promise.all([
          fetchJSON<{ ok: boolean; version: string }>("/api/health"),
          fetchJSON<{ stages: StageSummary[] }>("/api/stages"),
        ]);
        stages = list.stages;
        status = `connected · v${health.version}`;
      } catch (err) {
        status = `backend unreachable: ${err instanceof Error ? err.message : err}`;
      }
    })();
  });
</script>

<main>
  <h1>labeler</h1>
  <p class="status">{status}</p>

  {#if stages.length > 0}
    <ul class="stages">
      {#each stages as stage (stage.id)}
        <li>
          <strong>{stage.title}</strong>
          <span>{stage.description}</span>
        </li>
      {/each}
    </ul>
  {:else}
    <p class="empty">
      No workflow stages registered yet — the annotate and review stages land
      with the shared canvas engine.
    </p>
  {/if}
</main>
