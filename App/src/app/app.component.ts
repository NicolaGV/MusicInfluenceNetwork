// app.component.ts
import { Component } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';

import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  imports: [FormsModule, HttpClientModule, CommonModule]
})
export class AppComponent {
  artistName1: string | null = null;
  artistName2: string | null = null;

  influenceScore: number | null = null;
  errorMessage: string = '';

  graphUrl: SafeResourceUrl | null = null;
  pathLength: number | null = null;
  loading: boolean = false;

  graphType: string = 'influence';

  constructor(private http: HttpClient, private sanitizer: DomSanitizer) { }


  getInfluenceScore() {
    const payload = {
      artist_name_1: this.artistName1,
      artist_name_2: this.artistName2
    };

    this.http.post<any>('http://localhost:5000/influence-diffusion', payload)
      .subscribe({
        next: (response) => {
          this.influenceScore = response.influence_score;
          this.errorMessage = '';
        },
        error: (error) => {
          this.errorMessage = error.error?.error || 'An error occurred';
          this.influenceScore = null;
        }
      });
  }

  generateGraph() {
    this.loading = true;
    const payload = {
      artist_name_1: this.artistName1,
      artist_name_2: this.artistName2,
      graph_type: this.graphType
    };

    this.http.post<any>('http://localhost:5000/generate-graph', payload)
      .subscribe({
        next: (response) => {
          this.graphUrl = this.sanitizer.bypassSecurityTrustResourceUrl(response.graph_url);
          this.pathLength = response.path_length;
          this.errorMessage = '';
          this.loading = false;

        },
        error: (error) => {
          this.errorMessage = error.error?.error || 'An error occurred';
          this.graphUrl = null;
          this.pathLength = null;
          this.loading = false;
        }
      });
  }
}