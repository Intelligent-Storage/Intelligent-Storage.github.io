---
layout: default
title: Posts
permalink: /posts/
---

<div class="page">
  <h1 style="text-align: center; margin-bottom: 2rem;">✦ All Posts</h1>

  <div class="posts-list">
    {% for post in site.posts %}
    <article class="topic-post">
      <div class="topic-post-content">
        <h2 class="topic-post-title">
          <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
        </h2>
        {% if post.tags.size > 0 or post.categories.size > 0 %}
        <div class="topic-post-meta">
          {% if post.tags contains 'logic-scientific-method' %}
            <span class="post-category">Introduction to Logic and Scientific Method</span>
          {% elsif post.categories contains 'design-review' %}
            <span class="post-category">Design Review</span>
          {% else %}
            {% for tag in post.tags limit:1 %}
              <span class="post-category">{{ tag | replace: '-', ' ' | capitalize }}</span>
            {% endfor %}
          {% endif %}
        </div>
        {% endif %}
      </div>
      <time class="topic-post-date">{{ post.date | date: "%B %d, %Y" }}</time>
    </article>
    {% endfor %}
  </div>
</div>

<style>
  .posts-list {
    max-width: 800px;
    margin: 0 auto;
  }
  
  .topic-post {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 1rem 0;
    border-bottom: 1px dotted rgba(212, 165, 71, 0.4);
    transition: all 0.2s ease;
  }
  
  .topic-post:hover {
    padding-left: 1rem;
    background: rgba(212, 165, 71, 0.05);
  }
  
  .topic-post-content {
    flex: 1;
  }
  
  .topic-post-title {
    margin: 0 0 0.25rem 0;
    font-family: 'Work Sans', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
  }
  
  .topic-post-title a {
    color: var(--dark-olive);
    text-decoration: none;
    transition: color 0.3s ease;
  }
  
  .topic-post-title a:hover {
    color: var(--retro-pink);
  }
  
  .topic-post-meta {
    margin-top: 0.25rem;
  }
  
  .post-category {
    display: inline-block;
    font-size: 0.75rem;
    color: var(--retro-pink);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
  }
  
  .topic-post-date {
    font-size: 0.85rem;
    color: #999;
    margin-left: 2rem;
    white-space: nowrap;
  }
</style>
